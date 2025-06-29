# ⚡ **AICTE Internship - Week-2 Assignment**

## **Transfer Learning with EfficientNetV2B0 (Image Classification)**

---

## 📅 **Week 2: Model Building, Training, Evaluation & Visualization**

---

## 🔁 **1. Data Augmentation Pipeline**

Real-time augmentation improves robustness:

```python
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])
```

✅ **Purpose:**

* Handles variations in orientation
* Reduces overfitting
* Improves generalization

---

## 🏗️ **2. Model Architecture - EfficientNetV2B0 Transfer Learning**

### **Load Pretrained Model**

```python
base_model = tf.keras.applications.EfficientNetV2B0(
    input_shape=(128, 128, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False
```

---

### **Complete Model Assembly**

```python
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(128, 128, 3)),
    data_augmentation,
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

✅ **Highlights:**

* Utilizes pre-trained ImageNet knowledge
* Only later layers are trainable
* Added Dropout for regularization

---

## ⚙️ **3. Model Compilation & Training**

### **Compile Model**

```python
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['Accuracy'])
```

---

### **Early Stopping to Prevent Overfitting**

```python
early = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)
```

---

### **Train the Model**

```python
history = model.fit(
    datatrain,
    validation_data=datavalid,
    epochs=15,
    batch_size=100,
    callbacks=[early]
)
```

---

## 📊 **4. Training & Validation Metrics Visualization**

```python
acc = history.history['Accuracy']
val_acc = history.history['val_Accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(10, 8))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training vs Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training vs Validation Loss')

plt.show()
```

✅ **Purpose:**

* Visualizes performance over epochs
* Detects overfitting or underfitting trends

---

## 🧪 **5. Final Model Evaluation**

```python
loss, accuracy = model.evaluate(datatest)
print(f'Test accuracy is {accuracy:.4f}, Test loss is {loss:.4f}')
```

---

## 🧩 **6. Confusion Matrix & Classification Report**

### **Generate Predictions**

```python
y_true = np.concatenate([y.numpy() for x, y in datatest], axis=0)
y_pred_probs = model.predict(datatest)
y_pred = np.argmax(y_pred_probs, axis=1)
```

---

### **Evaluation Metrics**

```python
print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))
```

---

### **Heatmap Visualization**

```python
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=class_names,
            yticklabels=class_names,
            cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
```

✅ **Purpose:**

* Class-wise error inspection
* Visual understanding of prediction accuracy

---

## ✅ **Summary of Week 2**

* Applied **data augmentation** for better generalization
* Used **EfficientNetV2B0** for transfer learning
* Trained with **early stopping** to avoid overfitting
* Visualized accuracy, loss, and confusion matrix
* Evaluated performance with proper metrics

---

# 📝 **End of Week-2 Assignment**

---

