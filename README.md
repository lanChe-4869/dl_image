先安装环境，自行开一个虚拟python环境

```
pip install -r requirements.txt
```




# 图像下载教程
## 1. 获取dy的图片blog的 note_id

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
* 设置关键参数 modal_urls_file，用于读取上一步获得的note_id
* output_dir，设置图像下载保存文件夹
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
process\02_filter_delete_low_res_images.py
这个脚本用于筛出来图片大小较低的图片，设置好 source_dir 和 output_dir，以及 size_threshold，把图片小的直接筛走，默认500kb的图片会被筛走
* ```
  自行设置第6行 source_dir 和第8行 output_dir
  然后，直接运行：
  python process\02_filter_delete_low_res_images.py
  ```

process\03_divide_images.py
这个脚本使用 yolo 来检测人，我这里默认用了 yolo11n.pt，你们电脑性能好的话可以上大一点的模型，如 yolo11s.pt
该脚本主要将数据分成三部分：单人照、双人照、多人照。
* ```
  设置143行 source_dir 和144行 output_dir 的路径
  然后直接运行:
  python process\03_divide_images.py
  ```

  





