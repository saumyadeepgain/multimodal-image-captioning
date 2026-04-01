import torch
import torch.nn.functional as F

def generate_caption_greedy(model, image, vocab, max_length=50, device="cuda"):
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        # Extract features once
        features = model.encoder(image)
        
        # Start with <start> token
        caption = [vocab.stoi["<start>"]]
        attention_maps = []
        
        for _ in range(max_length):
            caption_tensor = torch.tensor(caption).unsqueeze(0).to(device)
            
            # Pass to decoder
            # Decoder returns (batch, seq_len, vocab_size) and attention weights
            outputs, attn_weights = model.decoder(features, caption_tensor, return_attention=True)
            
            # Get the prediction for the last token
            next_word_logits = outputs[0, -1, :]
            predicted_word_idx = next_word_logits.argmax(-1).item()
            
            caption.append(predicted_word_idx)
            # Store attention map for the generated word
            attention_maps.append(attn_weights[0, -1, :])
            
            if predicted_word_idx == vocab.stoi["<end>"]:
                break
                
    # Convert tokens back to words
    generated_words = [vocab.itos[idx] for idx in caption]
    return generated_words, attention_maps

def generate_caption_beam_search(model, image, vocab, max_length=50, beam_size=3, device="cuda"):
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        features = model.encoder(image)
        
        start_idx = vocab.stoi["<start>"]
        end_idx = vocab.stoi["<end>"]
        
        # List of tuples: (sequence, score)
        # sequence is a list of token indices
        beams = [([start_idx], 0.0)]
        completed_beams = []
        
        for _ in range(max_length):
            new_beams = []
            
            for seq, score in beams:
                if seq[-1] == end_idx:
                    completed_beams.append((seq, score))
                    continue
                    
                caption_tensor = torch.tensor(seq).unsqueeze(0).to(device)
                outputs = model.decoder(features, caption_tensor)
                
                next_word_logits = outputs[0, -1, :]
                # Use log softmax for numerical stability when adding probabilities
                next_word_probs = F.log_softmax(next_word_logits, dim=-1)
                
                # Get top k candidates
                top_probs, top_indices = next_word_probs.topk(beam_size)
                
                for i in range(beam_size):
                    new_seq = seq + [top_indices[i].item()]
                    new_score = score + top_probs[i].item()
                    new_beams.append((new_seq, new_score))
                    
            if not new_beams:
                break
                
            # Keep top k beams
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
            
        # Add any remaining incomplete beams
        completed_beams.extend(beams)
        # Sort by length-normalized score to avoid bias towards shorter sentences
        completed_beams = sorted(completed_beams, key=lambda x: x[1] / len(x[0]), reverse=True)
        
        best_seq = completed_beams[0][0]
        generated_words = [vocab.itos[idx] for idx in best_seq]
        
        # Note: In a full implementation, you'd store attention for each beam path.
        # For simplicity, if attention is needed for beam search, we rerun greedy on the best path,
        # or we just rely on greedy search for attention visualization.
        
        return generated_words
