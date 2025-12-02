
[*] Training model on 3 packets...
Traceback (most recent call last):
  File "/home/neamaabd/Study_Project/Assignment3/Task1/task1_sheet3.py", line 291, in <module>
    sheet3_task1()
  File "/home/neamaabd/Study_Project/Assignment3/Task1/task1_sheet3.py", line 257, in sheet3_task1
    labeled_data = load_and_label_data()
  File "/home/neamaabd/Study_Project/Assignment3/Task1/task1_sheet3.py", line 229, in load_and_label_data
    print(data_normal_arrays[0].shape)
AttributeError: 'list' object has no attribute 'shape'