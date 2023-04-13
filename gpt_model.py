import tensorflow as tf
from preprocess import vocab_size

class FeedForward(tf.keras.Model):
  def __init__(self, n_embd=768, num_heads=12, dropout_coef=0.3):
    super(FeedForward, self).__init__()
    self.net=tf.keras.Sequential([
        #TODO: check if this should be changed
        tf.keras.layers.Dense(num_heads*n_embd),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dense(n_embd),
        tf.keras.layers.Dropout(dropout_coef)
    ])
  def call(self, x):
    x=self.net(x)
    return x
  
class TransformerBlock(tf.keras.Model):
    def __init__(self,n_embd=768,num_heads=12):
        self.ln1=tf.keras.layers.LayerNormalization(),
        self.ff=FeedForward(n_embd=n_embd,num_heads=num_heads),
        self.ln2=tf.keras.layers.LayerNormalization(),
        self.sa=tf.keras.layers.MultiHeadAttention("TODO")
    def call(self,x):
        x=x+self.ff(self.ln1(x))
        x=x+self.sa(self.ln2(x))

class GPT2_small(tf.keras.Model):
    def __init__(self,num_transformer_blocks=12,n_embd=768,num_heads=12,context_len=200):
        #using default parameters for GPT2, but shortening context_len, since captions are fairly short in general
        #TODO: check caption size distribution and use longest as context len probably
        self.token_embedding_table=tf.keras.layers.Embedding(vocab_size, n_embd)
        self.position_embedding_table=tf.keras.layers.Embedding(context_len, n_embd)
        self.transformer_blocks=tf.keras.Sequential()
        for _ in range(num_transformer_blocks):
            self.transformer_blocks.add(TransformerBlock(n_embd=n_embd,num_heads=num_heads))
        self.layernorm_out=tf.keras.layers.LayerNormalization()
        self.feedforward_out=FeedForward(n_embd)
        #task is to map to embedding of size ~1000, leaving output dimension as n_embd
        #also leaving logits without softmax
    def call(self,x):
        tok_emb=self.token_embedding_table(x)
        pos_emb=self.position_embedding_table(tf.range(x.shape[-2])) #second dim from end are input positions
        x=tok_emb+pos_emb
        x=self.transformer_blocks(x)
        x=self.layernorm_out(x)
        x=self.feed_forward(x)
        return x
    
