# Dodo People Detection Test
# Table Cleaning Detection Prototype

## 🇷🇺 Русская версия

### 📌 Описание

Это упрощённый прототип системы анализа использования столов в ресторане (например, пиццерии) на основе видео.

Система определяет:

* когда стол становится пустым
* когда к нему подходит человек
* и считает время между этими событиями

---

### 🎯 Выбранное видео и столик

* Использовано видео: **Видео 1** (замени на своё)
* Один стол выбран вручную через `cv2.selectROI`
* Стол расположен в **(центре/слева/справа кадра)**

Пример ROI:

```
x = ..., y = ..., w = ..., h = ...
```

---

### 🧠 Логика решения

1. **Обработка кадров**

   * Видео обрабатывается покадрово через OpenCV

2. **Детекция людей**

   * Используется YOLOv8n
   * Учитываются только объекты класса `person`

3. **Определение "у стола"**

   * Если bbox человека пересекается с ROI — считаем, что он у стола

4. **Сглаживание**

   * Используются пороги:

     * `occupied_threshold = 8`
     * `empty_threshold = 12`

5. **Машина состояний**

   * `EMPTY` — стол пуст
   * `OCCUPIED` — стол занят
   * Фиксируются события:

     * `approach_detected`
     * `table_became_empty`

---

### 📊 Результаты

По обработанному видео:

👉 **Среднее время между уходом и следующим подходом: X.XX секунд**

---

### 📁 Выходные файлы

* `output.mp4` — видео с визуализацией
* `events.csv` — события
* `report.csv` — расчёты

---

### ⚠️ Ограничения

* Стол выбирается вручную
* Простая логика пересечения bbox
* Нет трекинга
* Нет разделения гостей и сотрудников
* Возможны ошибки детекции

---

### 🛠️ Возможные улучшения

* Добавить трекинг
* Использовать IoU или центр bbox
* Добавить поддержку нескольких столов
* Улучшить устойчивость

---

### 🖼️ Проблемный кадр

![Проблемный кадр](screenshots/problem_frame.png)

Описание:

* Частичное пересечение человека со столом
* Возможны ложные срабатывания

---

## 🚀 Запуск

```bash
pip install -r requirements.txt
python main.py --video video1.mp4
```

---

## 🇬🇧 English Version

### 📌 Overview

This project is a simplified prototype for detecting table usage events in a restaurant (e.g., a pizzeria) using video analysis.

The system detects:

* when a table becomes empty
* when a person approaches the table
* and calculates the delay between these events

The solution is designed as a lightweight and explainable pipeline without training custom models.

---

### 🎯 Selected Video and Table

* Video used: **Video 1** (replace with actual one)
* One table was selected manually using `cv2.selectROI`
* The selected table is clearly visible and located in the **(center/left/right part of the frame)**

Example ROI (after selection):

```
x = ..., y = ..., w = ..., h = ...
```

---

### 🧠 Detection Logic

The pipeline consists of the following steps:

1. **Frame Processing**

   * Video is processed frame-by-frame using OpenCV

2. **Person Detection**

   * YOLOv8n (Ultralytics) is used to detect people in each frame
   * Only objects of class `person` are considered

3. **Table Interaction Detection**

   * A person is considered "near the table" if their bounding box intersects with the selected ROI

4. **Temporal Smoothing**

   * To reduce noise, frame counters are used:

     * `occupied_threshold = 8`
     * `empty_threshold = 12`

5. **State Machine**

   * The system maintains two states:

     * `EMPTY`
     * `OCCUPIED`
   * Transitions generate events:

     * `approach_detected`
     * `table_became_empty`

---

### 📊 Results

From the processed video:

* Total events detected: **X** (optional)
* Average delay between table becoming empty and next approach:

👉 **Average delay: X.XX seconds**

---

### 📁 Output Files

* `output.mp4` — annotated video with visualization
* `events.csv` — raw event log
* `report.csv` — calculated delays

---

### ⚠️ Limitations

* ROI is selected manually (no automatic table detection)
* Intersection-based logic may produce false positives
* No distinction between staff and guests
* No object tracking (each frame processed independently)
* YOLO detection may occasionally miss or misdetect people

---

### 🛠️ Possible Improvements

* Add tracking (e.g., DeepSORT)
* Use IoU or center-based filtering instead of simple intersection
* Detect staff vs guests
* Support multiple tables
* Improve robustness with temporal models

---

### 🖼️ Problematic Case

Below is an example of a problematic frame:

![Problem frame](screenshots/problem_frame.png)

Description:

* A person is near the table but only partially intersects ROI
* This may lead to false "occupied" detection
