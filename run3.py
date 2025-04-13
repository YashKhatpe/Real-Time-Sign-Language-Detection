# import os
# import torch
# import numpy as np
# from torchvision import transforms
# import videotransforms
# from pytorch_i3d import InceptionI3d
# import cv2
# import json
# import time
# from tqdm import tqdm

# class SignLanguageRecognizer:
#     def _init_(self, config):
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.load_model(config['model_path'], config['num_classes'])
#         self.load_class_map(config['class_map_path'])
#         self.transforms = self.get_transforms()
#         self.num_frames = config.get('num_frames', 64)
#         self.stride = config.get('stride', 1)
#         self.threshold = config.get('confidence_threshold', 0.5)

#     def load_model(self, model_path, num_classes):
#         self.model = InceptionI3d(400, in_channels=3)
#         self.model.replace_logits(num_classes)
#         checkpoint = torch.load(model_path, map_location=self.device)
        
#         if 'state_dict' in checkpoint:
#             self.model.load_state_dict(checkpoint['state_dict'])
#         else:
#             self.model.load_state_dict(checkpoint)
            
#         self.model = self.model.to(self.device)
#         self.model.eval()
#         print(f"Loaded model from {model_path}")

#     def load_class_map(self, class_map_path):
#         with open(class_map_path) as f:
#             self.class_map = json.load(f)
#         self.idx_to_class = {v: k for k, v in self.class_map.items()}
#         print(f"Loaded class map with {len(self.class_map)} classes")

#     def get_transforms(self):
#         return transforms.Compose([
#             videotransforms.CenterCrop(224),
#             videotransforms.ClipToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])
#         ])

#     def preprocess_video(self, video_path):
#         # Read video frames
#         cap = cv2.VideoCapture(video_path)
#         frames = []
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frames.append(frame)
#         cap.release()

#         # Select frames with stride
#         total_frames = len(frames)
#         if total_frames < self.num_frames:
#             raise ValueError(f"Video too short: {total_frames} frames")
            
#         indices = np.linspace(0, total_frames-1, self.num_frames, dtype=int)
#         selected_frames = [frames[i] for i in indices]

#         # Apply transforms
#         transformed = self.transforms(selected_frames)
#         return transformed.unsqueeze(0)  # Add batch dimension

#     def predict(self, video_path):
#         try:
#             # Preprocess
#             inputs = self.preprocess_video(video_path).to(self.device)
            
#             # Inference
#             with torch.no_grad():
#                 start_time = time.time()
#                 outputs = self.model(inputs)
#                 outputs = outputs.mean(dim=2)  # Temporal pooling
#                 probs = torch.softmax(outputs, dim=1)
#                 inference_time = time.time() - start_time

#             # Get predictions
#             confidence, pred_idx = torch.max(probs, 1)
#             confidence = confidence.item()
#             pred_class = self.idx_to_class[pred_idx.item()]

#             return {
#                 'class': pred_class,
#                 'confidence': confidence,
#                 'inference_time': inference_time,
#                 'valid_prediction': confidence >= self.threshold
#             }
#         except Exception as e:
#             return {'error': str(e)}

#     def batch_predict(self, video_dir):
#         results = {}
#         video_files = [f for f in os.listdir(video_dir) 
#                       if f.split('.')[-1] in ['mp4', 'avi', 'mov']]
        
#         for video_file in tqdm(video_files):
#             video_path = os.path.join(video_dir, video_file)
#             results[video_file] = self.predict(video_path)
            
#         return results

# if __name__ == '__main__':
#     config = {
#         'model_path': '/content/drive/MyDrive/I3D/checkpoints/best_39class_checkpoint.pth',
#         'class_map_path': '/content/drive/MyDrive/I3D/preprocess/nslt_40.json',
#         'num_classes': 39,
#         'num_frames': 64,
#         'stride': 2,
#         'confidence_threshold': 0.6
#     }

#     recognizer = SignLanguageRecognizer(config)
    
#     # Single video prediction
#     test_video = '/path/to/test_video.mp4'
#     result = recognizer.predict(test_video)
#     print("\nPrediction Result:")
#     print(f"Class: {result['class']}")
#     print(f"Confidence: {result['confidence']:.2%}")
#     print(f"Inference Time: {result['inference_time']:.2f}s")
    
#     # Batch prediction
#     # video_dir = '/path/to/videos/'
#     # results = recognizer.batch_predict(video_dir)
#     # print("\nBatch Results:")
#     # for video, pred in results.items():
#     #     print(f"{video}: {pred['class']} ({pred['confidence']:.2%})")