from ultralytics import YOLO 

model = YOLO('models/best.pt')

results = model.predict('input_videos/A1606b0e6_0 (12).mp4',save=True)
print(results[0])

# print(results)
print('=====================================')
for box in results[0].boxes:
    print(box)

    # with open("users.txt", "a") as f:
    #     f.write(box + "\n")

# 0: ball, 1: goalkeeper, 2: players and 3: refrees