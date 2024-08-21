import os
import cv2
import torch
import random
import matplotlib.pyplot as plt


def separate_sample(model, transform, test_loader, threshold=0.005):
    mse = torch.nn.MSELoss()
    model.initialize_cell_and_hidden_state()
    idx = random.randint(0, len(test_loader)-1)
    dir = f'F:/test/KSPJM/test/{idx*5}'
    frames = sorted([frame for frame in os.listdir(dir)])

    x = []
    y = []
    avg = []
    window = 10
    print(f'Sample index: {idx}. Processing...')
    for i in range(len(frames) - 1):
        img = cv2.imread(os.path.join(dir, frames[i]))
        next = cv2.imread(os.path.join(dir, frames[i+1]))

        x = transform([img])
        x = torch.tensor(x[0], dtype=torch.float32)
        x = x.view(1, 133, 2)

        next = transform([next])
        next = torch.tensor(next[0], dtype=torch.float32)
        next = next.view(133, 2)
    
        outputs = model(x)
        outputs = outputs.view(outputs.shape[1] // 2, 2)
        loss = mse(outputs, next)

        y.append(loss.item())
        print(i)

    for i in range(len(frames) - 1):
        img = cv2.imread(os.path.join(dir, frames[i]))
        overlay = img.copy()
        overlay[:] = (0, 0, 255)
        cv2.addWeighted(overlay, y[i] / max(y), img, 1 - y[i]/max(y), 0, img)
    
        cv2.imshow("Video", img)
        key = cv2.waitKey(10)
        if key == ord('q'):
            break

    for frame in range(len(y)-window):
        x.append(frame)
        avg.append(sum(y[frame:frame+window]) / window)

    plt.plot(x, avg)
    plt.show()


def inference(model, label_map, transform):
    actions = dict([(value, key) for key, value in label_map.items()])
    window_width = 10
    tokens = ['' for _ in range(window_width)]
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot access camera.")
        exit()

    model.initialize_cell_and_hidden_state()
    action_text = ""

    while True:
        # Grab frame
        success, img = cap.read()
        if not success:
            print("Failed to capture image.")
            return False
        
        # Extract landmarks
        x = transform([img])
        x = torch.tensor(x[0], dtype=torch.float32)
        x = x.view(1, 133, 2)
        output = model(x)

        # Pass input through network
        output[0] = torch.nn.functional.softmax(output[0])
        confidence, predicted_index = torch.max(output, dim=1)
        predicted_action = actions[predicted_index.item()]

        # Output the recognized action
        action_text = f'{predicted_action}'
        tokens.append(action_text)
        tokens.pop(0)
        token = max(set(tokens), key=tokens.count)
        print(f'\r{tokens}', end='')
        cv2.putText(img, token, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, int(confidence * 255), int(255 - confidence * 255)), 2, cv2.LINE_AA)

        if confidence > 0.6:
            model.initialize_cell_and_hidden_state()
            print('\r'+ ' ' * 100, end='')
            print(f'\rRecognized action: {predicted_action} with confidence: {confidence.item():.2f}', end='')
        else:
            pass
            print('\r'+ ' ' * 100, end='')
            print(f'\rUnknown action. Most likely: {predicted_action} with confidence: {confidence.item():.2f}', end='')

        # Show image
        cv2.imshow('Camera', img)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def inference_optical_flow(model, actions, transform):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot access camera.")
        exit()

    model.initialize_cell_and_hidden_state()
    action_text = ""

    # Grab first frame so that there are 2 of them to process at the first inference step
    success, img = cap.read()
    if not success:
        print("Failed to capture image.")
        return False
    cv2.putText(img, action_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Camera', img)
    cv2.waitKey(1)

    prev = img

    while True:
        # Grab current frame
        success, img = cap.read()
        if not success:
            print("Failed to capture image.")
            return False
        cv2.putText(img, action_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Camera', img)

        curr = img

        # Extract optical flow
        flow = transform(prev, curr)

        # Pass input through network
        output = model(flow)
        output[0] = torch.nn.functional.softmax(output[0])
        confidence, predicted_index = torch.max(output, dim=1)
        predicted_action = actions[predicted_index.item()]

        # Output the recognized action
        if confidence > 0.6:
            action_text = f'{predicted_action}'
            print('\r'+ ' ' * 100, end='')
            print(f'\rRecognized action: {predicted_action} with confidence: {confidence.item():.2f}', end='')
        else:
            print('\r'+ ' ' * 100, end='')
            print(f'\rUnknown action. Most likely: {predicted_action} with confidence: {confidence.item():.2f}', end='')

        prev = curr

        # Show image
        cv2.imshow('Camera', img)
        cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()