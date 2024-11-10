from ultralytics import YOLO
model = YOLO('yolov8m.pt')
results = model.predict(r'C:\Users\user\Downloads\ForBrass2.jpg')
from PIL import Image
for i, r in enumerate(results):
	img_bgr = r.plot()
	im_rgb = Image.fromarray(img_bgr[...,::-1])
	r.show()
	r.save(filename = f"results{i}.jpg")