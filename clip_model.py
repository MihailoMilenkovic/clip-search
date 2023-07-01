import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from tensorflow.keras import regularizers
import wandb
import random

from dataset_loader import load_coco_dataset

class CLIP(tf.keras.Model):
  def __init__(self,embedding_size=1000,temperature=1.0,learning_rate=0.001,batch_size=2):
    super().__init__()
    #using BERT as text encoder
    self.language_preprocessor = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
    self.language_encoder=hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4",trainable=True)

    #using vision transformer as image encoder
    self.image_encoder=hub.KerasLayer("https://tfhub.dev/sayakpaul/vit_s16_fe/1", trainable=True)

    self.image_projection=tf.keras.layers.Dense(embedding_size,use_bias=False,kernel_regularizer=regularizers.L2(1.0))
    self.text_projection=tf.keras.layers.Dense(embedding_size,use_bias=False,kernel_regularizer=regularizers.L2(1.0))

    self.image_model=tf.keras.Sequential([
      self.image_encoder,
      self.image_projection,
    ])
    self.text_model=tf.keras.Sequential([
      self.language_preprocessor,
      self.language_encoder,
      self.text_projection
    ])
    self.temperature=tf.Variable(temperature)
    self.batch_size=batch_size
    self.optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
    self.text_model_path="models/CLIP_text"
    self.image_model_path="models/CLIP_image"

  def getImageEmbedding(self,image_batch):
    return self.image_model(image_batch)

  def getTextEmbedding(self,text_batch):
    return self.text_model(text_batch)
  
  @tf.function
  def train_batch(self,image_batch,text_batch):
    print("okay before image emb")
    image_emb=self.getImageEmbedding(image_batch)
    print("okay before text emb")
    text_emb=self.getTextEmbedding(text_batch)
    print("okay after text emb")
    logits=(image_emb*tf.transpose(text_emb))*tf.math.exp(self.temperature)
    labels=tf.eye(self.batch_size)
    loss_img=tf.keras.losses.CategoricalCrossentropy(axis=0)(labels,logits)
    loss_text=tf.keras.losses.CategoricalCrossentropy(axis=1)(labels,logits)
    loss=(loss_img+loss_text)/2
    return loss

  def train_loop(self,num_epochs=1):
    coco_dataset=load_coco_dataset(split="train", batch_size=self.batch_size, preprocessing_type="training")
    # TODO: include after debugging
    # wandb.init()
    cnt=0
    for epoch in range(num_epochs):
      for (image_batch,caption_batch) in coco_dataset:
        cnt+=1
        #select 1 random caption from each example for current training iteration
        text_batch=tf.gather(caption_batch, tf.cast(tf.random.uniform([1])*caption_batch.shape[1], tf.int32), axis=1)
        print("curr text batch:",text_batch)
        with tf.GradientTape() as tape:
          loss=self.train_batch(self,image_batch=image_batch,text_batch=text_batch) 
        #TODO: log this when training loop is working
        # wandb.log({"loss":loss.numpy()})

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        #TODO: remove later, just debugging for now
        if cnt%7==0:
          break
        if cnt%5000==50:
          # TODO: evaluate CLIP score on subset of training data over time and calculate correlation coefficient with our model
          # can be done using https://torchmetrics.readthedocs.io/en/stable/multimodal/clip_score.html
          # scores should be increasingly correlated over time
          # note: CLIP score is calculated as max(100*cos(e_i,e_c),0) where e_i and e_c are image and caption embeddings of our model
          tf.keras.saving.save_model(self.text_model, f"{self.text_model_path}_{cnt}_batches")
          tf.keras.saving.save_model(self.image_model, f"{self.image_model_path}_{cnt}_batches")
      break
    tf.keras.saving.save_model(self.text_model, self.text_model_path)
    tf.keras.saving.save_model(self.image_model, self.image_model_path)
  
  def load_model(self):
    self.model=tf.keras.saving.load_model(self.model_path)
          

if __name__=="__main__":
  model=CLIP()
  model.train_loop(num_epochs=1)