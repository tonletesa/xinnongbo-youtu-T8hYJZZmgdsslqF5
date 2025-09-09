[合集 - 深度学习(6)](https://github.com)

1.基于yolo12对目标物体进行自动裁剪和模糊打码09-09

[2.15分钟速通yolo12，从环境搭建到推理图片，最后训练自己的数据集08-22](https://github.com/codingtea/p/19053136)[3.如何用labelimg标注yolo数据集，并利用工具自动划分数据集08-26](https://github.com/codingtea/p/19059273)[4.通过matlab训练和验证深度学习的目标检测08-28](https://github.com/codingtea/p/19062895):[nuts坚果](https://tidati.com)[5.谁说的YOLO只能目标检测？手把手教你解锁它隐藏的热力图视野！09-02](https://github.com/codingtea/p/19070568)[6.【YOLOv12实战】寥寥数行代码实现目标跟踪与速度估计，新手也能轻松搞定！09-04](https://github.com/codingtea/p/19073884)

收起

​

## 视频演示

[基于yolo12对目标物体进行自动裁剪和模糊打码](https://github.com)

---

## 引言

本篇文章将深入剖析一个基于 YOLO12 模型的视频处理工具的代码实现逻辑与核心功能。该工具能够对视频中的目标物体（如行李箱）进行自动裁剪和模糊打码处理，适用于隐私保护、目标提取等场景。我们将重点讲解代码的核心逻辑、YOLO12 模型的应用以及实现效果，简要提及工具的使用方式。

![gif](https://img2024.cnblogs.com/blog/3687401/202509/3687401-20250909095701015-1261382756.gif)

![主界面](https://img2024.cnblogs.com/blog/3687401/202509/3687401-20250909095736656-296191858.png)

---

## 核心功能与效果

该工具通过 **Ultralytics YOLO** 框架，利用 YOLO12 模型实现以下核心功能：

1. **目标物体裁剪**：自动检测视频中的目标物体（例如行李箱），并将其裁剪为单独的图片保存。
2. **目标物体模糊**：对检测到的目标物体区域进行模糊打码处理，并在模糊区域显示物体类别和置信度。
3. **实时处理与保存**：支持实时处理视频流，保存裁剪图片和处理后的视频。

效果展示：

* **输入**：一段机场传送带运输行李箱的视频。
* **输出**：

  + 裁剪后的行李箱图片，保存至指定目录。
  + 模糊打码后的视频，行李箱区域被模糊处理，并标注类别和置信度。
  + 用户可通过界面控制播放、暂停和保存操作。

![裁剪效果]()

---

## 代码实现逻辑

工具的核心代码分为两个主要类：VideoProcessor（负责视频处理逻辑）和 VideoPlayer（负责用户界面和交互）。以下从逻辑和实现的角度详细分析。

### 1. VideoProcessor 类：核心处理逻辑

VideoProcessor 类封装了视频加载、帧处理和结果保存的逻辑，基于 YOLO12 模型实现目标检测、裁剪和模糊功能。

#### 1.1 初始化与模型配置

```
def __init__(self):
    self.cap = None
    self.current_frame = None
    self.output_dir = "output"
    self.crop_enabled = False
    self.blur_enabled = False
    self.processed_frames = []
    # 初始化 YOLOv12 模型
    self.cropper = solutions.ObjectCropper(model="yolo12n.pt", show=False)
    self.blurrer = solutions.ObjectBlurrer(model="yolo12n.pt", show=False)
    if not os.path.exists(self.output_dir):
        os.makedirs(self.output_dir)
```

![]()

* **YOLO12 模型**：使用 Ultralytics 的 solutions 模块，初始化 ObjectCropper 和 ObjectBlurrer 两个对象，分别用于裁剪和模糊处理，均基于预训练模型 yolo12n.pt。
* **输出目录**：默认创建 output 目录，用于保存裁剪图片和处理后的视频。
* **状态管理**：通过 crop\_enabled 和 blur\_enabled 控制是否启用裁剪或模糊功能。

#### 1.2 视频加载与帧获取

```
def load_video(self, video_path):
    if self.cap:
        self.cap.release()
    self.cap = cv2.VideoCapture(video_path)
    return self.cap.isOpened()

def get_frame(self):
    if self.cap and self.cap.isOpened():
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame
            return frame
    return None
```

![]()

* **视频加载**：使用 OpenCV 的 cv2.VideoCapture 加载视频文件，确保视频有效性。
* **帧获取**：逐帧读取视频，保存当前帧到 current\_frame。

#### 1.3 帧处理（裁剪与模糊）

```
def process_frame(self, frame):
    processed = frame.copy()
    if self.crop_enabled:
        results = self.cropper(frame)
        processed = results.plot_im if hasattr(results, 'plot_im') else processed
    if self.blur_enabled:
        results = self.blurrer(frame)
        processed = results.plot_im if hasattr(results, 'plot_im') else processed
    self.processed_frames.append(processed)
    return processed
```

![]()

* **核心逻辑**：

  + **裁剪**：调用 ObjectCropper 处理视频帧，自动检测目标物体并裁剪，保存裁剪结果到 output 目录。
  + **模糊**：调用 ObjectBlurrer 对目标物体区域进行模糊处理，标注类别和置信度。
* **结果存储**：处理后的帧存储在 processed\_frames 列表中，用于后续视频保存。
* **灵活性**：通过 crop\_enabled 和 blur\_enabled 动态控制是否应用裁剪或模糊。

#### 1.4 保存处理结果

```
def save_frame(self, frame, filename):
    filepath = os.path.join(self.output_dir, filename)
    cv2.imwrite(filepath, frame)
    return filepath
```

![]()

* **裁剪图片保存**：将裁剪后的图片保存到指定目录。
* **视频保存**（在 VideoPlayer 中实现，见下文）。

### 2. VideoPlayer 类：用户界面与交互

VideoPlayer 类基于 PyQt5 实现图形界面，负责视频显示、用户交互和处理控制。

#### 2.1 界面初始化

```
def init_ui(self):
    central_widget = QWidget()
    self.setCentralWidget(central_widget)
    main_layout = QHBoxLayout(central_widget)
    left_panel = QVBoxLayout()
    self.video_label = QLabel()
    self.video_label.setAlignment(Qt.AlignCenter)
    self.video_label.setMinimumSize(640, 480)
    left_panel.addWidget(self.video_label)
    # 控制按钮
    self.select_btn = QPushButton("选择视频")
    self.play_btn = QPushButton("播放")
    self.run_btn = QPushButton("运行处理")
    self.save_btn = QPushButton("保存视频")
    self.open_dir_btn = QPushButton("打开输出目录")
    # 处理选项
    self.crop_check = QCheckBox("裁剪")
    self.blur_check = QCheckBox("模糊")
    # 输出目录选择
    self.output_btn = QPushButton("选择输出目录")
    self.output_dir_label = QLabel("当前输出目录: " + self.processor.output_dir)
```

![]()

* **界面布局**：

  + 视频显示区域：QLabel 用于显示视频帧。
  + 控制按钮：包括“选择视频”“播放”“运行处理”“保存视频”和“打开输出目录”。
  + 处理选项：通过复选框控制裁剪和模糊功能。
  + 输出目录：支持用户自定义输出路径。

#### 2.2 视频播放与处理

```
def update_frame(self):
    frame = self.processor.get_frame()
    if frame is not None:
        if self.is_processing:
            processed_frame = self.processor.process_frame(frame)
            self.show_frame(processed_frame)
        else:
            self.show_frame(frame)
    else:
        self.timer.stop()
        self.playing = False
        self.play_btn.setText("播放")
        self.processor.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.is_processing = False
```

![]()

* **实时播放**：通过 QTimer 每 30ms 更新一帧（约 30fps）。
* **处理控制**：当 is\_processing 为 True 时，调用 process\_frame 进行裁剪或模糊处理。
* **帧显示**：将 OpenCV 帧转换为 Qt 图像，缩放后显示在 video\_label 上。

#### 2.3 保存处理后的视频

```
def save_video(self):
    if not self.processor.processed_frames:
        QMessageBox.warning(self, "错误", "请先运行处理视频")
        return
    file_path, _ = QFileDialog.getSaveFileName(self, "保存视频", "", "视频文件 (*.mp4 *.avi)")
    if file_path:
        frame_width = int(self.processor.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.processor.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.processor.cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(file_path, fourcc, fps, (frame_width, frame_height))
        for frame in self.processor.processed_frames:
            out.write(frame)
        out.release()
        QMessageBox.information(self, "成功", "视频保存完成！")
```

![]()

* **视频保存**：使用 OpenCV 的 VideoWriter 将处理后的帧合成为视频，保存为用户指定的文件名（支持 .mp4 和 .avi 格式）。
* **参数获取**：从原始视频中获取帧率、分辨率等参数，确保输出视频一致。

---

## 技术亮点

1. **YOLO12 模型高效性**：

   * 利用 yolo12n.pt 预训练模型，实现高效的目标检测、裁剪和模糊。
   * ObjectCropper 和 ObjectBlurrer 提供便捷的 API，简化开发流程。
2. **实时处理**：

   * 通过定时器和帧处理机制，实现视频流的实时处理与显示。
   * 支持动态切换裁剪和模糊功能，灵活性强。
3. **用户友好性**：

   * PyQt5 界面直观，支持视频预览、播放控制和结果保存。
   * 提供输出目录选择和文件浏览功能，方便用户管理结果。
4. **模块化设计**：

   * VideoProcessor 和 VideoPlayer 分离，逻辑清晰，便于维护和扩展。

---

## 总结

这个基于 YOLO12 的视频处理工具通过 Ultralytics 的 solutions 模块，实现了目标物体的自动裁剪和模糊打码功能。核心逻辑包括视频加载、帧处理（裁剪与模糊）、结果保存和用户交互，效果显著且操作简便。代码结构清晰，模块化设计使其易于扩展，适合视频处理、隐私保护等场景。

希望这篇文章帮助你理解 YOLO12 视频处理工具的实现逻辑！欢迎在评论区讨论或分享你的优化建议

​
