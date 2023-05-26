import tensorflow as tf
import tensorflow_hub as hub

from dataset_loader import load_coco_dataset

class CLIP(tf.keras.Model):
  def __init__(self,embedding_size=1000,temperature=1.0,learning_rate=0.001,batch_size=8):
    #using BERT as text encoder
    self.language_preprocessor = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
    self.language_encoder=hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4",trainable=True)

    #using vision transformer as image encoder
    self.image_encoder=hub.KerasLayer("https://tfhub.dev/sayakpaul/vit_s16_fe/1", trainable=True)

    self.image_projection=tf.keras.layers.Dense(embedding_size,use_bias=False)
    self.text_projection=tf.keras.layers.Dense(embedding_size,use_bias=False)

    self.temperature=tf.Variable(temperature)
    self.batch_size=batch_size
    self.optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
    self.model_path="models/CLIP"

  def getImageEmbedding(self,image_batch):
    #image preprocessing already done while loading dataset
    image_encoded=self.image_encoder(image_batch)
    image_emb=self.image_projection(image_encoded)
    image_emb_normalized=tf.math.l2_normalize(image_emb)
    return image_emb_normalized

  def getTextEmbedding(self,text_batch):
    text_preprocessed=self.language_preprocessor(text_batch)
    text_encoded=self.language_encoder(text_preprocessed)
    text_emb=text_encoded["pooled_output"] #TODO: check this
    text_emb_normalized=tf.math.l2_normalize(text_emb)
    return text_emb_normalized
  
  @tf.function
  def train_batch(self,image_batch,text_batch):
    image_emb=self.getImageEmbedding(image_batch)
    text_emb=self.getTextEmbedding(text_batch)
    logits=(image_emb*tf.transpose(text_emb))*tf.math.exp(self.temperature)
    labels=tf.eye(self.batch_size)
    loss_img=tf.keras.losses.CategoricalCrossentropy(axis=0)(labels,logits)
    loss_text=tf.keras.losses.CategoricalCrossentropy(axis=1)(labels,logits)
    loss=(loss_img+loss_text)/2
    return loss

  @tf.function
  def train_loop(self,num_batches):
    coco_dataset=load_coco_dataset(num_batches,self.batch_size)
    for (batch,idx) in zip(coco_dataset,range(num_batches)):
      image_batch, text_batch=batch
      with tf.GradientTape() as tape:
        loss=self.train_batch(self,image_batch,text_batch) 
      grads = tape.gradient(loss, self.trainable_weights)
      self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

      if idx%5000==0:
        tf.keras.saving.save_model(self.model, f"{self.model_path}_{idx}")

    tf.keras.saving.save_model(self.model, self.model_path)
          

if __name__=="__main__":
  
  model=CLIP()
  model.train_loop(num_batches=10)#change to around 50000 for full training