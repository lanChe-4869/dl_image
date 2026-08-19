python ./process/04_image_crop_seg_v3_fail_origin_debug.py `
  --input_dir ./images/test_100 `
  --output_dir ./images/process_image/test_100 `
  --model yolo11s-seg.pt `
  --conf 0.35 `
  --padding 0.25 `
  --top_k 2 `
  --min_people 2

python ./process/04_image_crop_seg_v3_fail_origin_debug.py `
  --input_dir ./images/process_image/test_100/debug/process_fail/one_person `
  --output_dir ./images/process_image/test_100 `
  --model yolo11s-seg.pt `
  --conf 0.1 `
  --padding 0.25 `
  --top_k 2 `
  --min_people 2
  
  --person_duplicate_iou 1.1 `
  --person_duplicate_cover 1.1