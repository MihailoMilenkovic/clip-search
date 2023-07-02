import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from tensorflow.keras import regularizers
import wandb
import random

from dataset_loader import load_coco_dataset

class NormalizationLayer(tf.keras.layers.Layer):
  def call(self, inputs):
      norm = tf.norm(inputs, axis=-1, keepdims=True)
      normalized_inputs = inputs / norm
      return normalized_inputs

class CLIP(tf.keras.Model):
  def __init__(self,embedding_size=256,temperature=1.0,learning_rate=1e-5,batch_size=8):
    super().__init__()
    self.batch_size=batch_size

    # self.image_encoder=hub.KerasLayer("https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/3", trainable=True)
    self.image_encoder= hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4", output_shape=[1280],trainable=True)
    self.image_projection=tf.keras.layers.Dense(embedding_size,use_bias=False)
    self.text_projection=tf.keras.layers.Dense(embedding_size,use_bias=False)

    self.image_model=tf.keras.Sequential([
      self.image_encoder,
      self.image_projection,
      NormalizationLayer()
    ])

    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
    text_preprocessor = hub.KerasLayer(
    "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
    encoder_inputs = text_preprocessor(text_input)
    #using small bert model because large one takes too much compute
    text_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-128_A-2/2",trainable=True)
    # text_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-256_A-4",trainable=True)
    pooled_output=text_encoder(encoder_inputs)["pooled_output"]
    self.text_model= tf.keras.Sequential([
      tf.keras.Model(text_input, pooled_output),
      self.text_projection,
      NormalizationLayer()
    ])

    self.temperature=tf.Variable(temperature)
    self.optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
    self.text_model_path="models/CLIP_text"
    self.image_model_path="models/CLIP_image"

  def get_image_embedding(self,image_batch):
    return self.image_model(image_batch)

  def get_text_embedding(self,text_batch):
    return self.text_model(text_batch)

  @tf.function(reduce_retracing=True)
  def train_batch(self,image_batch,text_batch):
    image_emb=self.get_image_embedding(image_batch)
    text_emb=self.get_text_embedding(text_batch)
    # print("image emb:",image_emb)
    # print("text emb:",text_emb)
    logits=tf.matmul(image_emb,text_emb,transpose_b=True)*tf.exp(self.temperature)
    # print("logits:",logits)
    labels=tf.eye(self.batch_size)
    loss_img=tf.keras.losses.CategoricalCrossentropy(axis=0)(labels,logits)
    loss_text=tf.keras.losses.CategoricalCrossentropy(axis=1)(labels,logits)
    loss=(loss_img+loss_text)/2
    return loss

  def get_random_captions(self,caption_batch):
    return tf.constant(tf.gather(caption_batch, tf.cast(tf.random.uniform([1])*caption_batch.shape[1], tf.int32), axis=1))

  def train_loop(self,num_epochs=1):
    coco_dataset=load_coco_dataset(split="train", batch_size=self.batch_size, preprocessing_type="training")
    wandb.init()
    cnt=0
    for epoch in range(num_epochs):
      for (image_batch,caption_batch) in coco_dataset:
        cnt+=1
        #select 1 random caption from each example for current training iteration
        text_batch=self.get_random_captions(caption_batch)
        with tf.GradientTape() as tape:
          loss=self.train_batch(image_batch=image_batch,text_batch=text_batch) 
        wandb.log({"loss":loss.numpy()})

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        #TODO: remove if training working over first couple of batches
        if cnt%200==0:
          break

        if cnt%5000==0:
          # TODO: evaluate CLIP score on subset of training data over time and calculate correlation coefficient with our model
          # can be done using https://torchmetrics.readthedocs.io/en/stable/multimodal/clip_score.html
          # scores should be increasingly correlated over time
          # note: CLIP score is calculated as max(100*cos(e_i,e_c),0) where e_i and e_c are image and caption embeddings of our model
          tf.keras.saving.save_model(self.text_model, f"{self.text_model_path}_{cnt}_batches")
          tf.keras.saving.save_model(self.image_model, f"{self.image_model_path}_{cnt}_batches")
    tf.keras.saving.save_model(self.text_model, self.text_model_path)
    tf.keras.saving.save_model(self.image_model, self.image_model_path)
  
  def load_model(self):
    self.text_model=tf.keras.saving.load_model(self.text_model_path)
    self.image_model=tf.keras.saving.load_model(self.image_model_path)
          

if __name__=="__main__":
  # tf.config.run_functions_eagerly(True)  # Temporarily run eagerly for easier debugging
  model=CLIP()
  model.train_loop(num_epochs=1)