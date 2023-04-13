import tensorflow as tf
import numpy as np
from gpt_model import GPT2
from resnet_model import ResNet50
from preprocess import get_batch_annotated_images

model_path="models/CLIP"

class CLIP(tf.keras.Model):
  def __init__(self,embedding_size=1000):
    self.gpt=GPT2()
    self.resnet=ResNet50()
    self.image_projection=tf.keras.layers.Dense(embedding_size)
    self.text_projection=tf.keras.layers.Dense(embedding_size)
    self.temperature=tf.Variable(shape=(1,))
    self.optimizer=tf.keras.optimizers.Adam()
  def getImageEmbedding(self,image_batch):
    image_outputs=ResNet50(image_batch)
    image_projected=self.image_projection(image_outputs)
    image_normalized=tf.math.l2_normalize(image_projected)
    return image_normalized
  def getTextEmbedding(self,text_batch):
    text_outputs=GPT2(text_batch)
    text_projected=self.text_projection(text_outputs)
    text_normalized=tf.math.l2_normalize(text_projected)
    return text_normalized
  @tf.function
  def train_batch(self,image_batch,text_batch):
    n=len(image_batch)
    image_outs=self.getImageEmbedding(image_batch)
    text_outs=self.getImageEmbedding(text_batch)
    logits=(text_outs*tf.transpose(image_outs))*tf.math.exp(self.temperature)
    labels=tf.eye(n)
    loss_img=tf.keras.losses.CategoricalCrossentropy(axis=0)(labels,logits)
    loss_text=tf.keras.losses.CategoricalCrossentropy(axis=1)(labels,logits)
    loss=(loss_img+loss_text)/2
    return loss
  @tf.function
  def train_loop(self,num_batches,batch_size=8):
    for i in range(num_batches):
      image_batch,text_batch=get_batch_annotated_images(batch_size)
      with tf.GradientTape() as tape:
        loss=self.train_batch(self,image_batch,text_batch) 
      grads = tape.gradient(loss, self.trainable_weights)
      self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

      if i%5000==0:
        tf.keras.saving.save_model(self.model, model_path)
    
    tf.keras.saving.save_model(self.model, model_path)
          

if __name__=="__main__":
  num_batches=10#500000
  model=CLIP()
  model.train_loop(num_batches)