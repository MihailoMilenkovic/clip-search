import tensorflow as tf

class Conv2DBatchNormReLU(tf.keras.Model):
  def __init__(self, filters=64, kernel_size=7, strides=1, padding="same"):
    self.model=tf.keras.Sequential([
      tf.keras.layers.Conv2DBatchNormReLU(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding),
      tf.keras.layers.BatchNormalization(),
      tf.keras.activations.ReLU()
    ])
  def call(self,x):
    return self.model(x)

class ResNet50(tf.keras.Model):
  def __init__(self):
    self.conv1=tf.keras.layers.Conv2DBatchNormReLU(filters=64, kernel_size=7, strides=2)
    self.max_pool=tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=2)
    self.conv_blocks=[]
    for i in range(3):
      conv2_block=tf.keras.Sequential([
        Conv2DBatchNormReLU(filters=64, kernel_size=1),
        Conv2DBatchNormReLU(filters=64, kernel_size=3),
        Conv2DBatchNormReLU(filters=256, kernel_size=1)
      ])
      self.conv_blocks.append(conv2_block)
    for i in range(4):
      conv3_block=tf.keras.Sequential([
        Conv2DBatchNormReLU(filters=128, kernel_size=1, strides=2 if i==0 else 1),
        Conv2DBatchNormReLU(filters=128, kernel_size=3),
        Conv2DBatchNormReLU(filters=512, kernel_size=1)
      ])
      self.conv_blocks.append(conv3_block)
    for i in range(6):
      conv4_block=tf.keras.Sequential([
        Conv2DBatchNormReLU(filters=256, kernel_size=1, strides=2 if i==0 else 1),
        Conv2DBatchNormReLU(filters=256, kernel_size=3),
        Conv2DBatchNormReLU(filters=1024, kernel_size=1)
      ])
      self.conv_blocks.append(conv4_block)
    for i in range(3):
      conv5_block=tf.keras.Sequential([
        Conv2DBatchNormReLU(filters=512, kernel_size=1, strides=2 if i==0 else 1),
        Conv2DBatchNormReLU(filters=512, kernel_size=3),
        Conv2DBatchNormReLU(filters=2048, kernel_size=1)
      ])
      self.conv_blocks.append(conv5_block)
    #TODO: check if this is how it should be pooled
    self.avg_pool=tf.keras.layers.AveragePooling2D(pool_size=(2,2), strides=None, padding="same")
    self.dense=tf.keras.layers.Dense(1000,activation="softmax")

  def call(self,x):
    x=self.conv1(x)
    x=self.max_pool(x)
    for conv_block in self.conv_blocks:
      x=x+conv_block(x)
    x=self.avg_pool(x)
    x=self.dense(x)
    return x

model=ResNet50()
print(model.summary)