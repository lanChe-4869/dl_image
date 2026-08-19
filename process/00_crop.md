debug/process_fail文件夹里的几种情况:
no_person            YOLO 一个 person 都没识别到
one_person           只识别/确认到 1 个主体人物
duplicate_person     原始有多个 person 框，但去重后不足 2 人
small_second_person  第二个人太小，被判断成背景路人
bad_aspect           两个主体识别成功，但允许比例无法安全裁切
invalid_crop         裁切框非法
other_fail           其他未归类失败

2. failed_no_feasible_allowed_aspect_ratio:
没有合适的裁切比例, 调整padding:
--padding 0.18
降低底部留白：
--bottom_padding_multiplier 1.3


3. failed_not_enough_unique_people_unique0_raw0_unique0: 2
YOLO 完全没有检测到人, 降低置信度：
--conf 0.25


4. failed_not_enough_unique_people_unique1_raw1_unique1: 16
有 16 张图 YOLO 只检测到 1 个人, 第二个主体漏检:
降低置信度：--conf 0.25


5. failed_not_enough_unique_people_unique1_raw2_unique1: 6
有 6 张图 YOLO 原始检测到了 2 个 person 框，但去重后只剩 1 个唯一人物。
可以放宽去重：
--person_duplicate_iou 0.70
--person_duplicate_cover 0.92