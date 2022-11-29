from tensorflow.keras import layers, models, Model, Sequential


def AlexNet_v1(im_height=224, im_width=224, num_classes=1000):
    # tensorflow 中的 tensor 通道排序是 NHWC
    # 深度为 3 数据类型为 float 32
    input_image = layers.Input(shape=(im_height, im_width, 3), dtype="float32")  # output(None, 224, 224, 3)
    x = layers.ZeroPadding2D(((1, 2), (1, 2)))(input_image)  # output(None, 227, 227, 3)
    x = layers.Conv2D(48, kernel_size=11, strides=4, activation="relu")(x)  # output(None, 55, 55, 48)
    x = layers.MaxPool2D(pool_size=3, strides=2)(x)  # output(None, 27, 27, 48)
    x = layers.Conv2D(128, kernel_size=5, padding="same", activation="relu")(x)  # output(None, 27, 27, 128)
    x = layers.MaxPool2D(pool_size=3, strides=2)(x)  # output(None, 13, 13, 128)
    x = layers.Conv2D(192, kernel_size=3, padding="same", activation="relu")(x)  # output(None, 13, 13, 192)
    x = layers.Conv2D(192, kernel_size=3, padding="same", activation="relu")(x)  # output(None, 13, 13, 192)
    x = layers.Conv2D(128, kernel_size=3, padding="same", activation="relu")(x)  # output(None, 13, 13, 128)
    x = layers.MaxPool2D(pool_size=3, strides=2)(x)  # output(None, 6, 6, 128)

    # 展成一维向量
    x = layers.Flatten()(x)  # output(None, 6*6*128)
    # 按 0.2 比例随机失活神经元
    x = layers.Dropout(0.2)(x)
    # 全连接层 1
    x = layers.Dense(2048, activation="relu")(x)  # output(None, 2048)
    x = layers.Dropout(0.2)(x)
    # 全连接层 2
    x = layers.Dense(2048, activation="relu")(x)  # output(None, 2048)
    # 全连接层 3
    x = layers.Dense(num_classes)(x)  # output(None, 5)
    predict = layers.Softmax()(x)

    # 定义网络的输入和输出(得到的概率分布)
    model = models.Model(inputs=input_image, outputs=predict)
    return model


class AlexNet_v2(Model):
    def __init__(self, num_classes=1000):
        super(AlexNet_v2, self).__init__()
        self.features = Sequential([
            layers.ZeroPadding2D(((1, 2), (1, 2))),  # output(None, 227, 227, 3)
            layers.Conv2D(48, kernel_size=11, strides=4, activation="relu"),  # output(None, 55, 55, 48)
            layers.MaxPool2D(pool_size=3, strides=2),  # output(None, 27, 27, 48)
            layers.Conv2D(128, kernel_size=5, padding="same", activation="relu"),  # output(None, 27, 27, 128)
            layers.MaxPool2D(pool_size=3, strides=2),  # output(None, 13, 13, 128)
            layers.Conv2D(192, kernel_size=3, padding="same", activation="relu"),  # output(None, 13, 13, 192)
            layers.Conv2D(192, kernel_size=3, padding="same", activation="relu"),  # output(None, 13, 13, 192)
            layers.Conv2D(128, kernel_size=3, padding="same", activation="relu"),  # output(None, 13, 13, 128)
            layers.MaxPool2D(pool_size=3, strides=2)])  # output(None, 6, 6, 128)

        self.flatten = layers.Flatten()
        self.classifier = Sequential([
            layers.Dropout(0.2),
            layers.Dense(1024, activation="relu"),  # output(None, 2048)
            layers.Dropout(0.2),
            layers.Dense(128, activation="relu"),  # output(None, 2048)
            layers.Dense(num_classes),  # output(None, 5)
            layers.Softmax()
        ])

    def call(self, inputs, **kwargs):
        x = self.features(inputs)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
