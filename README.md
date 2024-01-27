# CNN-projects

CIFAR-10 veri setini kullanarak 60.00 görsel alacağız bu görsellerle CNN modeli eğiteceğiz. Aşağıda tedaylı şekilde açıklamaya çalıştım.


### Öncelikle CNN'in nasıl çalıştığını anlamaya çalışalım.

Temel olarak, Cnn, sınıflandırma sorununun çözümü için standart Sinir Ağı kullanır, ancak bilgileri belirlemek ve bazı özellikleri tespit etmek için diğer katmanları kullanır.


### CNN'in ana mantığı

![cnn](https://github.com/whasancan/CNN_projects/blob/d60fbaf3d0fe8fd70ba9c01489e7ff66ea78c0ac/foto/cnnnnn.png)


## Bu kütüphaneleri kullancağız.

```python 
import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
```


## CIFAR-10 veri setini kullacağız demiştik, resimlerimizi indirelim, eğitim ve test verilerini ayrı ayrı değişkenlere atıyalım.

```python 
# veri setini indirelim 60.000resim vardır
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
```


## Verilerin piksel değerini 0-1 arasına sıkı8ştırarak normalize eder. Bu, genellikle piksel değerlerini daha küçük bir aralığa getirerek modelin daha iyi öğrenmesine yardımcı olabilir.

```python 
#piksel değerini 0 ile 1 arasına sıkıştıralım
train_images, test_images = train_images / 255, test_images / 255
```


## Eğitim veri setinden 25 görüntüyü ve bunların sınıf isimlerini çekerek görsel sonuç oluşturup gösterir.

```python 
# verileri doğrulamak için ilk 25 görüntüyü ve isimlerini çekelim
class_names = ["airplane","automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck" ]

plt.figure(figsize=(7,7))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()
```

![örnek](https://github.com/whasancan/CNN_projects/blob/8613cba51a49b9e40016fc5cd9b43cb7335bbd22/foto/veri_resim.png)


## Burada CNN modeli oluşturup Sequential modelini tanımlıyoruz. Model Conv2D ve MaxPooling2D katmanlarını içerir. Bu katmanlar, tipik bir evrişimli sinir ağı mimarisini oluşturur ve evrişim ve havuzlama (pooling) işlemleriyle özellik haritalarını çıkarır.


```python 
# evrişimli katmanımızı oluşturalım
# girdi olarak  (image_height, image_width, color_channels) boyurunda resim alır
# color_channels (R,G,B) anlamına gelir

model = models.Sequential()    # model oluşturuluyor

# Conv2D katmanı oluşturuyoruz. (3,3) boyutunda 32 filitre kullanıyor. input_shape olarak 32x32  boyutunda 3renk katmanlı giriş alıyor
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))

# MaxPooling2D katmanını ekler. gelen özellik haritasından en büyük değeri seçer ve küçültür.
# bu katman önemli özellikleri vurgulamaya ve hesaplma maaliyetin azaltmaya yardımcı olur.
model.add(layers.MaxPooling2D((2,2)))

# ikinci Conv2D katmanını ekliyor
model.add(layers.Conv2D(64, (3,3), activation = "relu"))

# ikinci MaxPooling2D katmanını ekliyoruz
model.add(layers.MaxPooling2D((2, 2)))

# üçüncü Conv2D katmanını ekliyoruz. 64 filitre kuolanıcak
model.add(layers.Conv2D(64, (3,3), activation= "relu"))

print("Katmanlar oluşturuldu!")
```

## Bu kısımı resim ile anlatmak gerekirse, mavili kısım burda kod olarak yaptığımız kısmın görsel halidir.

![conv](https://github.com/whasancan/CNN_projects/blob/d60fbaf3d0fe8fd70ba9c01489e7ff66ea78c0ac/foto/conv_poolling.jpg)


## Oluşturulan modelin mimarisini özetleyelim. Her katmandaki parametre sayısını ve toplam parametre sayısını gösterir.

```python 
# modelimizin mimarisine bakalım
model.summary()
```

![özet1](https://github.com/whasancan/CNN_projects/blob/5ef7db0bf94d5e0b85f2ea0e0bb23bd8b187e750/foto/ilk_summary.png)


## Modelin sonuna yoğun katman ekler. Bu katman. önceki evrişimli katmanların çıkışlarını düzleştirir ve ardından tam bağlantılı(DENSE) katmamları ekler.

```python 
# Yoğuun katman oluşturlaım

# bu katman yukarıdan gelen özellik haritasını düzleştirir. tek boyutlu vektöre dönüştürür.
model.add(layers.Flatten())

# düzenlenmiş veriyi alan 64 nöron içeren tam bağlantılı(DENSE) katmanı ekler. burada daha yüksek seviyeli özellikleri öğrenir
model.add(layers.Dense(64, activation="relu"))

# kullandığımız veri 10 çıkışlı olduğu için 10 çıktılı yoğun katman oluşturacağız
model.add(layers.Dense(10))
```

## Bu katmanın ne olduğunu birde görsel olarak görelim. Kırmızılı olan kısım Fully Connected(Yoğun Katamn)dır.

![yoğun](https://github.com/whasancan/CNN_projects/blob/5ef7db0bf94d5e0b85f2ea0e0bb23bd8b187e750/foto/yo%C4%9Fun%20katman.jpg)


## Tam olark oluşturluan modelin mimarisi özetler ve bu özet bilgileri ekrana yazdırır. Modelin iç yapısını anlamak ve eğitilecek parametrelerin sayısını görmek için bu fonksiyonu kullanabiliriz.

```python 
# modelimizin tam mimarisine bakalım
model.summary()
```

![özet2](https://github.com/whasancan/CNN_projects/blob/5ef7db0bf94d5e0b85f2ea0e0bb23bd8b187e750/foto/2.summaray.png)


## Bu kod bloğu, modeli derler (compile) ve eğitir (fit). Bu kod bloğu, modelin eğitim sürecini başlatır ve her epoch sonunda modelin performansını gözlemlemek için eğitim ve doğrulama veri setlerindeki kayıp ve doğruluk değerlerini kaydeder.

```python 
# şimid modelimiz derleyelim(COMPİLE) ve eğitelim(FIT)


# Adam, gradient descent optimizasyon algoritmasının geliştirilmiş bir versiyonudur.
# kullanım amacı modelin performansını atrırarak daga hızlı öğrenmesini sağlamaktır.

# kayıp fonksiyonunu belirliyoruz
model.compile(optimizer="adam",
             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])

# train_images (GÖRÜNTÜLER) ve train_labels(GÖRÜNTÜLERİN İSİMLERİ) modelin eğitim verilerini temsil eder
# epochs, modelin kaç epoch boyunca eğitileceğini belirler. bir epoch modeleni tüm eğitim verilerini birekz geçmesi anlamın gelir.
# validation, modelin her epoch sonunda test veri seti üzerindfen performansı değerlendirmeye yarar
history = model.fit(train_images, train_labels, epochs=50,
                   validation_data=(test_images, test_labels))

```

## eğitimi tamamladık

![eğitim](https://github.com/whasancan/CNN_projects/blob/5ef7db0bf94d5e0b85f2ea0e0bb23bd8b187e750/foto/e%C4%9Fitim_ft.png)


## Modelin eğitim sürecinde elde edilen doğruluk değerlerini çizdirir ve daha sonra test veri seti üzerinde modelin kayıp (loss) ve doğruluk (accuracy) değerlerini değerlendirir.

```python 
# eğitim boyunca modelin doğruluk(ACCURACY) değerini çizdirir
# ve daha sonra test veri setinden modelin kayıp(LOSS) ve doğruluk değerini değerlendirir


plt.plot(history.history['accuracy'], label='accuracy') # eher epoch'taki eğitim setinden elde edilen doğruluk değeri
plt.plot(history.history['val_accuracy'], label = 'val_accuracy') # her epoc'taki doğrulama setinden elde edilen değruluk değeri

# grafik ayarları:
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()


# modeli değerlendirme
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
```

![grafik](https://github.com/whasancan/CNN_projects/blob/5ef7db0bf94d5e0b85f2ea0e0bb23bd8b187e750/foto/grafik.png)


## Son olarak baaşrı oranına bakalım

```python 
print(f"Test Accuracy: {test_acc * 100:.2f}%")
```

![başarı](https://github.com/whasancan/CNN_projects/blob/5ef7db0bf94d5e0b85f2ea0e0bb23bd8b187e750/foto/ba%C5%9Far%C4%B1_son.png)
