import torch
import torch.nn as nn

# Positional Encoding module to inject sequence order awareness into transformer inputs
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Step 1: Compute positional encoding matrix values using sine and cosine functions
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Step 2: Add positional encoding values element-wise to input token embeddings
        x = x + self.pe[:, :x.size(1)]
        return x


# Main Transformer Decoder architecture adjusted for EfficientNet-B4 feature dimension (1792) with built-in .fit() and .predict() functions
class TransformerImageCaptioning(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_decoder_layers=4, dim_feedforward=2048, max_length=40, feature_input_shape=1792, dropout=0.1):
        super(TransformerImageCaptioning, self).__init__()
        
        # Step 1: Linear layer to project image feature vector size to transformer model hidden dimension
        self.feature_proj = nn.Linear(feature_input_shape, d_model)
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_length)
        
        # Step 2: Configure multi-head transformer decoder layers stack
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # Step 3: Final linear projection layer mapping hidden states to vocabulary token logits
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
        self.max_length = max_length

    def generate_square_subsequent_mask(self, sz, device):
        # Step 4: Generate causal triangular mask to prevent transformer decoder from seeing future tokens
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, image_features, captions):
        device = image_features.device
        batch_size = image_features.size(0)
        seq_len = captions.size(1)

        # Step 5: Project image features to serve as memory tokens for decoder attention blocks
        img_emb = self.feature_proj(image_features).unsqueeze(1)
        # Model ke forward method ke andar:

        # Pehle: img_emb shape -> [batch_size, hidden_dim] ya [batch_size, 1, 1, hidden_dim]
        if img_emb.dim() == 4:
            img_emb = img_emb.squeeze(1).squeeze(1) # Extra dimensions ko hataiye -> [batch_size, hidden_dim]
        
        if img_emb.dim() == 2:
            img_emb = img_emb.unsqueeze(1) # Seq len dimension add karein -> [batch_size, 1, hidden_dim]
            
        # Step 6: Embed text token sequences and add positional encodings
        cap_emb = self.embedding(captions) * math.sqrt(self.d_model)
        cap_emb = self.pos_encoder(cap_emb)
        
        # Step 7: Create target causal masks and padding masks
        tgt_mask = self.generate_square_subsequent_mask(seq_len, device)
        padding_mask = (captions == 0)

        # Step 8: Pass inputs through the transformer decoder stack
        output = self.transformer_decoder(
            tgt=cap_emb,
            memory=img_emb,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=padding_mask
        )
        
        # Step 9: Compute final output prediction logits across the entire vocabulary
        logits = self.fc_out(output)
        return logits

    # Built-in .fit() method accepting all necessary training parameters
    def fit(self, dataloader, epochs, vocab, device, save_path="transformer_image_captioning_effnet.pth"):
        # Step 1: Set model to training mode and transfer to target execution device
        self.to(device)
        self.train()
        
        vocab_size = len(vocab)
        
        # Step 2: Configure cross-entropy loss function ignoring padding indices and setup Adam optimizer
        criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<pad>"])
        optimizer = optim.Adam(self.parameters(), lr=3e-4)
        
        # Step 3: Configure learning rate scheduler to reduce learning rate upon plateau
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
        
        # Step 4: Loop through training epochs
        for epoch in range(epochs):
            total_loss = 0
            loop = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{epochs}]")
            
            for (img_features, captions) in loop:
                img_features = img_features.to(device)
                captions = captions.to(device)
                
                # Step 5: Split caption tensor into teacher-forcing input and target sequences
                captions_input = captions[:, :-1]
                captions_target = captions[:, 1:]
                
                optimizer.zero_grad()
                
                # Step 6: Forward pass to obtain prediction logits
                outputs = self(img_features, captions_input)
                
                # Step 7: Compute cross-entropy loss against targets
                loss = criterion(outputs.reshape(-1, vocab_size), captions_target.reshape(-1))
                
                # Step 8: Backpropagate gradients and step optimizer
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                loop.set_postfix(loss=loss.item())
                
            avg_loss = total_loss / len(dataloader)
            scheduler.step(avg_loss)
            print(f"Epoch [{epoch+1}/{epochs}] Completed. Average Loss: {avg_loss:.4f}")

        # Step 9: Save trained model weights to disk
        torch.save(self.state_dict(), save_path)
        print(f"Model training complete and weights saved successfully to {save_path}.")

    # Built-in .predict() method to take image feature input and generate a caption via Beam Search
    def predict(self, img_feature, vocab, device, beam_width=3):
        self.eval()
        img_features = img_feature.unsqueeze(0).to(device)

        start_token = vocab.stoi["<start>"]
        end_token = vocab.stoi["<end>"]
        
        # Step 1: Initialize beam search list with initial start token tuple
        beams = [(0.0, [start_token])]
        
        # Step 2: Generate tokens iteratively up to maximum length constraint
        for _ in range(self.max_length):
            all_candidates = []
            
            for score, seq in beams:
                if seq[-1] == end_token:
                    all_candidates.append((score, seq))
                    continue
                    
                captions_tensor = torch.tensor([seq], dtype=torch.long).to(device)
                
                with torch.no_grad():
                    outputs = self(img_features, captions_tensor)
                    
                log_probs = torch.log_softmax(outputs[:, -1, :], dim=-1)
                topk_log_probs, topk_indices = torch.topk(log_probs, beam_width, dim=-1)
                
                for i in range(beam_width):
                    token_id = topk_indices[0, i].item()
                    token_score = topk_log_probs[0, i].item()
                    all_candidates.append((score + token_score, seq + [token_id]))
                    
            ordered = sorted(all_candidates, key=lambda tup: tup[0], reverse=True)
            beams = ordered[:beam_width]
            
            if all(seq[-1] == end_token for _, seq in beams):
                break
                
        best_seq = beams[0][1]
        words = [vocab.itos.get(idx, "<unk>") for idx in best_seq[1:] if idx != end_token]
        return words
