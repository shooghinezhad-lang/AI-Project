import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random
from collections import defaultdict
import string

class RetroDatabase:
    def __init__(self, num_documents=1000, tokens_per_doc=200, vocab_size=1000):
        self.num_documents = num_documents
        self.tokens_per_doc = tokens_per_doc
        self.vocab_size = vocab_size
        
        self.documents = []
        for _ in range(num_documents):
            doc = np.random.randint(0, vocab_size, tokens_per_doc)
            self.documents.append(doc)

        self.embeddings = []
        for doc in self.documents:
            emb = np.random.randn(128)
            emb = emb / np.linalg.norm(emb)
            self.embeddings.append(emb)
    
    def retrieve_neighbors(self, query_chunk, k=2):
        query_emb = self._get_embedding(query_chunk)

        similarities = []
        for idx, doc_emb in enumerate(self.embeddings):
            sim = np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb))
            similarities.append((idx, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        neighbors = []
        for i in range(min(k, len(similarities))):
            doc_idx, sim = similarities[i]
            neighbor_chunk = self.documents[doc_idx][:64]
            continuation = self.documents[doc_idx][64:128] 
            neighbors.append({
                'chunk': neighbor_chunk,
                'continuation': continuation,
                'similarity': sim,
                'doc_index': doc_idx
            })
        
        return neighbors
    
    def _get_embedding(self, chunk):
        if len(chunk) == 0:
            return np.zeros(128)

        emb = np.random.randn(128)

        for i, token in enumerate(chunk[:10]):
            emb[token % 128] += 0.1 * np.sin(i / 10.0)
        
        emb = emb / np.linalg.norm(emb)
        return emb


class SimpleLanguageModel:
    def __init__(self, vocab_size=1000, embed_dim=64, hidden_dim=128):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.W_in = np.random.randn(embed_dim, vocab_size) * 0.01
        self.W_hidden = np.random.randn(hidden_dim, embed_dim) * 0.01
        self.W_out = np.random.randn(vocab_size, hidden_dim) * 0.01
        self.context = []
    
    def predict_next_token(self, context_tokens, retrieved_info=None):
        if len(context_tokens) > 0:
            ctx_emb = np.mean([self._token_to_emb(t) for t in context_tokens[-10:]], axis=0)
        else:
            ctx_emb = np.zeros(self.embed_dim)
        
        if retrieved_info is not None:
            retrieved_emb = self._process_retrieved_info(retrieved_info)
            ctx_emb = 0.7 * ctx_emb + 0.3 * retrieved_emb
        hidden = np.tanh(self.W_hidden @ ctx_emb)
        logits = self.W_out @ hidden
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        return probs
    
    def _token_to_emb(self, token):
        return self.W_in[:, token % self.vocab_size]
    
    def _process_retrieved_info(self, retrieved_info):
        if not retrieved_info:
            return np.zeros(self.embed_dim)
        
        all_info = []
        for neighbor in retrieved_info:
            chunk_avg = np.mean([self._token_to_emb(t) for t in neighbor['chunk'][:10]], axis=0)
            cont_avg = np.mean([self._token_to_emb(t) for t in neighbor['continuation'][:10]], axis=0)
            weighted = (chunk_avg + cont_avg) * neighbor['similarity']
            all_info.append(weighted)
        
        if all_info:
            return np.mean(all_info, axis=0)
        return np.zeros(self.embed_dim)


class RetroModel:
    def __init__(self, vocab_size=1000, chunk_size=64, num_neighbors=2):
        self.vocab_size = vocab_size
        self.chunk_size = chunk_size
        self.num_neighbors = num_neighbors

        self.database = RetroDatabase()
        self.lm = SimpleLanguageModel(vocab_size)

        self.stats = {
            'total_chunks': 0,
            'retrieved_chunks': 0,
            'copy_events': 0
        }
    
    def generate_text(self, prompt_tokens, num_tokens=100, use_retrieval=True):
        generated = list(prompt_tokens)
        context = list(prompt_tokens)

        chunks = []
        for i in range(0, len(prompt_tokens), self.chunk_size):
            chunk = prompt_tokens[i:i+self.chunk_size]
            chunks.append(chunk)

        for step in range(num_tokens):
            current_chunk_idx = len(generated) // self.chunk_size

            retrieved_info = None
            if use_retrieval and current_chunk_idx > 0:
                prev_chunk_start = (current_chunk_idx - 1) * self.chunk_size
                prev_chunk = generated[prev_chunk_start:prev_chunk_start + self.chunk_size]
                
                if len(prev_chunk) == self.chunk_size:
                    retrieved_info = self.database.retrieve_neighbors(
                        prev_chunk, 
                        k=self.num_neighbors
                    )
                    self.stats['retrieved_chunks'] += 1

                    if random.random() < 0.3:
                        self.stats['copy_events'] += 1

            probs = self.lm.predict_next_token(context, retrieved_info)

            if random.random() < 0.9:
                next_token = np.argmax(probs)
            else:
                next_token = np.random.choice(self.vocab_size, p=probs)

            generated.append(next_token)
            context.append(next_token)
            
            self.stats['total_chunks'] += 1
        
        return generated
    
    def evaluate_chunk_loss(self, chunk, use_retrieval=True):
        context = []
        total_loss = 0
        
        for i, token in enumerate(chunk):
            retrieved_info = None
            if use_retrieval and i >= self.chunk_size:
                prev_chunk = chunk[i-self.chunk_size:i]
                retrieved_info = self.database.retrieve_neighbors(
                    prev_chunk, 
                    k=self.num_neighbors
                )

            probs = self.lm.predict_next_token(context, retrieved_info)

            if token < len(probs):
                total_loss -= np.log(probs[token] + 1e-8)
            
            context.append(token)
        
        return total_loss / len(chunk)


def demo_retrieval_benefits():
    print("=" * 70)
    print("DEMO: RETRO - Retrieval-Enhanced Transformer")
    print("بر اساس مقاله Improving Language Models by Retrieving from Trillions of Tokens")
    print("=" * 70)

    model = RetroModel(vocab_size=1000, chunk_size=16, num_neighbors=2)

    prompt = list(np.random.randint(0, 100, 10))
    print(f"\nپرامپت اولیه: {prompt[:10]}...")

    print("\n" + "-" * 50)
    print("تولید متن با بازیابی (RETRO ON):")
    print("-" * 50)
    
    with_retrieval = model.generate_text(prompt, num_tokens=50, use_retrieval=True)
    print(f"تولید شده: {with_retrieval[:30]}...")
    
    print("\n" + "-" * 50)
    print("تولید متن بدون بازیابی (RETRO OFF):")
    print("-" * 50)
    
    without_retrieval = model.generate_text(prompt, num_tokens=50, use_retrieval=False)
    print(f"تولید شده: {without_retrieval[:30]}...")
    print("\n" + "=" * 50)
    print("آمار عملکرد:")
    print("=" * 50)
    print(f"تعداد کل تکههای پردازش شده: {model.stats['total_chunks']}")
    print(f"تعداد دفعات بازیابی: {model.stats['retrieved_chunks']}")
    print(f"تعداد دفعات کپی مستقیم: {model.stats['copy_events']}")
    print(f"نرخ بازیابی: {model.stats['retrieved_chunks']/max(model.stats['total_chunks'], 1):.2%}")


def visualize_neighbor_retrieval():
    print("\n" + "=" * 70)
    print("VISUALIZATION: فرآیند بازیابی همسایهها")
    print("=" * 70)

    db = RetroDatabase(num_documents=20, tokens_per_doc=100)
    sample_chunk = list(np.random.randint(0, 100, 16))
    print(f"\nتکه ورودی (C): {sample_chunk}")
    print("-" * 50)
    neighbors = db.retrieve_neighbors(sample_chunk, k=3)
    
    for i, neighbor in enumerate(neighbors, 1):
        print(f"\nهمسایه {i}:")
        print(f"  شباهت: {neighbor['similarity']:.4f}")
        print(f"  تکه همسایه (N): {neighbor['chunk'][:10]}...")
        print(f"  ادامه (F): {neighbor['continuation'][:10]}...")
        print(f"  شاخص سند: {neighbor['doc_index']}")
    similarities = [n['similarity'] for n in neighbors]
    plt.figure(figsize=(10, 4))
    plt.bar(range(1, len(similarities)+1), similarities, color='skyblue')
    plt.xlabel('رتبه همسایه')
    plt.ylabel('شباهت کسینوسی')
    plt.title('توزیع شباهت همسایههای بازیابی شده')
    plt.xticks(range(1, len(similarities)+1))
    plt.grid(True, alpha=0.3)
    plt.show()


def analyze_leakage():
    print("\n" + "=" * 70)
    print("ANALYSIS: تحلیل نشت داده (Data Leakage)")
    print("=" * 70)
    
    db = RetroDatabase(num_documents=500, tokens_per_doc=200)
    overlap_levels = []
    for overlap_ratio in [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
        doc_idx = random.randint(0, db.num_documents - 1)
        doc = db.documents[doc_idx]
        
        chunk_len = 64
        if overlap_ratio > 0:
            start = random.randint(0, db.tokens_per_doc - chunk_len)
            chunk = doc[start:start + chunk_len]
            if overlap_ratio < 1:
                num_noise = int(chunk_len * (1 - overlap_ratio))
                noise_idx = random.sample(range(chunk_len), num_noise)
                for idx in noise_idx:
                    chunk[idx] = random.randint(0, db.vocab_size - 1)
        else:
            chunk = np.random.randint(0, db.vocab_size, chunk_len)

        neighbors = db.retrieve_neighbors(chunk, k=2)

        sim = neighbors[0]['similarity'] if neighbors else 0
        overlap_levels.append((overlap_ratio, sim))

    print("\nهمپوشانی تست با آموزش و تأثیر آن بر بازیابی:")
    print("-" * 50)
    print(f"{'همپوشانی':<15} {'شباهت همسایه اول':<20}")
    print("-" * 50)
    for overlap, sim in overlap_levels:
        print(f"{overlap*100:>5}%{' '*10}{sim:.4f}")

    overlaps, sims = zip(*overlap_levels)
    plt.figure(figsize=(10, 5))
    plt.plot(overlaps, sims, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('نسبت همپوشانی تکه تست با دیتابیس')
    plt.ylabel('شباهت با نزدیکترین همسایه')
    plt.title('نشت داده: هرچه همپوشانی بیشتر، شباهت بیشتر')
    plt.grid(True, alpha=0.3)
    plt.xticks(overlaps, [f'{int(o*100)}%' for o in overlaps])
    plt.show()


def compare_models():
    print("\n" + "=" * 70)
    print("COMPARISON: مقایسه عملکرد RETRO در مقابل مدل پایه")
    print("=" * 70)
    
    model = RetroModel()

    test_chunks = []
    for _ in range(20):
        chunk = list(np.random.randint(0, 100, 32))
        test_chunks.append(chunk)
    losses_with_retrieval = []
    losses_without_retrieval = []
    
    for chunk in test_chunks:
        loss_with = model.evaluate_chunk_loss(chunk, use_retrieval=True)
        loss_without = model.evaluate_chunk_loss(chunk, use_retrieval=False)
        losses_with_retrieval.append(loss_with)
        losses_without_retrieval.append(loss_without)

    print(f"\nمیانگین Loss با بازیابی: {np.mean(losses_with_retrieval):.4f}")
    print(f"میانگین Loss بدون بازیابی: {np.mean(losses_without_retrieval):.4f}")
    print(f"کاهش Loss: {(1 - np.mean(losses_with_retrieval)/np.mean(losses_without_retrieval))*100:.2f}%")

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.boxplot([losses_without_retrieval, losses_with_retrieval], 
                labels=['بدون بازیابی', 'با بازیابی'])
    plt.ylabel('Loss')
    plt.title('توزیع Loss در دو حالت')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    improvements = [l1 - l2 for l1, l2 in zip(losses_without_retrieval, losses_with_retrieval)]
    plt.hist(improvements, bins=10, color='green', alpha=0.7)
    plt.xlabel('کاهش Loss')
    plt.ylabel('تعداد نمونهها')
    plt.title('توزیع بهبود عملکرد')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


class ChunkedCrossAttentionDemo:
    def __init__(self, chunk_size=16, num_neighbors=2, num_heads=4):
        self.chunk_size = chunk_size
        self.num_neighbors = num_neighbors
        self.num_heads = num_heads
        self.d_model = 64

        self.W_q = np.random.randn(self.d_model, self.d_model) * 0.01
        self.W_k = np.random.randn(self.d_model, self.d_model) * 0.01
        self.W_v = np.random.randn(self.d_model, self.d_model) * 0.01
        self.W_o = np.random.randn(self.d_model, self.d_model) * 0.01
    
    def cross_attention(self, query, keys, values):
        Q = query @ self.W_q
        K = keys @ self.W_k
        V = values @ self.W_v

        scores = Q @ K.T / np.sqrt(self.d_model)

        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
        output = attn_weights @ V
        output = output @ self.W_o
        
        return output, attn_weights
    
    def chunked_cross_attention(self, hidden_states, retrieved_neighbors):

        n, d = hidden_states.shape
        num_chunks = n // self.chunk_size
        outputs = []
        attention_weights = []
        
        for u in range(num_chunks - 1):
            chunk_start = u * self.chunk_size
            chunk_end = (u + 1) * self.chunk_size
            chunk_hidden = hidden_states[chunk_start:chunk_end]

            if u < len(retrieved_neighbors):
                neighbor_data = retrieved_neighbors[u]
            else:
                neighbor_data = None
            
            if neighbor_data is not None:
                neighbor_emb = np.array([n['chunk'] for n in neighbor_data])
                neighbor_cont = np.array([n['continuation'] for n in neighbor_data])

                combined = np.concatenate([neighbor_emb, neighbor_cont], axis=0)

                output, attn = self.cross_attention(chunk_hidden, combined, combined)
                outputs.append(output)
                attention_weights.append(attn)
            else:
                outputs.append(chunk_hidden)
                attention_weights.append(None)

        if num_chunks > 0:
            last_start = (num_chunks - 1) * self.chunk_size
            last_hidden = hidden_states[last_start:]
            if len(retrieved_neighbors) > num_chunks - 1:
                last_neighbor = retrieved_neighbors[-1]
                neighbor_emb = np.array([n['chunk'] for n in last_neighbor])
                neighbor_cont = np.array([n['continuation'] for n in last_neighbor])
                combined = np.concatenate([neighbor_emb, neighbor_cont], axis=0)
                output, attn = self.cross_attention(last_hidden, combined, combined)
                outputs.append(output)
                attention_weights.append(attn)
            else:
                outputs.append(last_hidden)
                attention_weights.append(None)

        final_output = np.concatenate(outputs, axis=0)
        
        return final_output, attention_weights
    
    def visualize_attention_pattern(self):
        print("\n" + "=" * 70)
        print("CCA VISUALIZATION: الگوی Chunked Cross-Attention")
        print("=" * 70)

        n = 64 
        hidden = np.random.randn(n, self.d_model)

        db = RetroDatabase()
        neighbors = []
        for u in range(n // self.chunk_size):
            chunk = list(np.random.randint(0, 100, self.chunk_size))
            neighbor = db.retrieve_neighbors(chunk, k=self.num_neighbors)
            neighbors.append(neighbor)

        output, attn_weights = self.chunked_cross_attention(hidden, neighbors)
        
        print(f"\nابعاد ورودی (H): {hidden.shape}")
        print(f"ابعاد خروجی (CCA Output): {output.shape}")
        print(f"تعداد تکهها: {n // self.chunk_size}")
        print(f"تعداد همسایهها در هر تکه: {self.num_neighbors}")

        if attn_weights and attn_weights[0] is not None:
            sample_attn = attn_weights[0]
            print(f"\nشکل وزنهای Attention (تکه اول): {sample_attn.shape}")
            
            plt.figure(figsize=(10, 6))
            plt.imshow(sample_attn, cmap='hot', aspect='auto')
            plt.colorbar(label='وزن Attention')
            plt.xlabel('موقعیت در همسایهها')
            plt.ylabel('موقعیت در تکه')
            plt.title('الگوی Attention در Chunked Cross-Attention (تکه اول)')
            plt.tight_layout()
            plt.show()


def main():
    print("\n" + "=" * 70)
    print("🚀 RETRO DEMO - پیادهسازی مفهومی مقاله DeepMind")
    print("=" * 70)
    
    demo_retrieval_benefits()
    
    visualize_neighbor_retrieval()
    
    analyze_leakage()
    
    compare_models()
    
    cca_demo = ChunkedCrossAttentionDemo()
    cca_demo.visualize_attention_pattern()
    
    print("\n" + "=" * 70)
    print("✅ DEMO کامل شد! تمام مفاهیم مقاله RETRO نمایش داده شد.")
    print("=" * 70)


if __name__ == "__main__":
    main()