先安装环境，自行开一个虚拟python环境

```
pip install -r requirements.txt
```




# douyin 图像下载教程
## 1. 获取douyin的图片blog的 note_id

* 运行 05_manual_click_collect_douyin_model_ids.py

  * 例如运行：

  * ```
    python 05_manual_click_collect_douyin_model_ids.py `
      --keyword 西湖 双人合照 `
      --output_txt "用于存放当前点击了哪些图片id" `
      --processed_ids_txt all_processed_modal_ids.txt
    ```

  * processed_ids_txt 里存放了我以前获得过的数据id，避免重复

* 上述代码运行后，先登录自己的douyin账号，然后可以自己改关键词搜索，点击筛选->选择 图文，其他的筛选自己看着来~~

* 点击了一个卡片之后，可以回去查看命令行有没有保存

* 照片尽可能先小窗预览看一下有没有合适的双人照，主要获取全身照、半身照



## 2. 下载对应图片

* 运行 06_download_douyin_note_images_fixed.py

  * ```
    python 05_download_douyin_note_images_fixed.py `
      --modal_urls_file "上一操作中，用于存放图片id的 txt 文件" `
      --output_dir images\double\20260819 `
      --target_count 5000
    ```

  * output_dir 设置下载保存路径，target_count 用于限制下载多少个 note_id 的图片，可以先不限制往大了写

* 下载过程中会打开浏览器它自己下载，中途浏览器可能会频繁跳出来，可以切新的桌面干别的事~~



## 3. 初步筛选

下载好图片后，里面会混杂一些黑图和表情包，可以先筛选一波

* ```
  process\02_filter_delete_low_res_images.py
  这个脚本用于筛出来图片大小较低的图片，设置好 source_dir 和 output_dir，以及 size_threshold，把图片小的直接筛走，默认500kb的图片会被筛走
  ```

* ```
  process\03_divide_images.py
  这个脚本使用 yolo 来，我这里默认用了 yolo11n.pt，你们电脑性能好的话可以上大一点的模型，如 yolo11s.pt
  主要将数据分成三个部分：单人照、双人照、多人照、others
  others包括：背景图、拼图、自拍的单人、双人等等。
  others这个需要你们自己筛了，yolo只能初步筛出来但不准。
  ```

  





