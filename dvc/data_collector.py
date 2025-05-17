import os
import numpy as np
import mediapipe as mp
from pynput import keyboard

from utils import *

ACTION_TO_IDX = {
    'czesc': 0,
    'dzien dobry': 1,
    'dobry wieczor': 2,
    'jak sie czujesz': 3,
    'dobrze': 4,
    'do widzenia': 6,
    'dobranoc': 7,
    'ja': 8,
    'ty': 9,
    'on_ona': 10,
    'oni': 11,
    'my': 12,
    'mam': 13,
    'nazwisko': 14,
    'imie': 15,
    'migowy znak': 16,
    'gluchy': 17,
    'slyszacy': 18,
    'masz': 19,
    'jest': 20,
    'poznac': 22,
    'mezczyzna': 25,
    'dzieci': 27,
    'kobieta': 29,
    'przepraszac': 32,
    'szkola': 34,
    'blank': 37,
    'byc': 38,
    'poznac': 39,
    'nosic': 40,
    'urodzic dziecko': 42,
    'zaprosic na': 43,
    'kochac': 44,
    'zostac': 45,
    'calowac': 46,
    'dorastac': 47,
    'kupic': 48,
    'umrzec': 49,
    'mowic': 50,
    'mieszkac': 51,
    'wstac': 52,
    'wziac prysznic': 53,
    'ubrac sie': 54,
    'lubic': 55,
    'nie lubic': 56,
    'wolec': 57,
    'odpoczywac': 58,
    'jesc': 59,
    'pic': 60,
    'wyjsc z domu': 61,
    'isc': 62,
    'wrocic do domu': 63,
    'sprzatac': 64,
    'isc spac': 65,
    'spac': 66,
    'pisac': 67,
    'napisac': 68,
    'podpisac': 69,
    'pomagac': 70,
    'migac': 71,
    'lat': 75,
    'kobieta': 77,
    'osoba': 79,
    'numer': 80,
    'telefon': 81,
    'zawod': 82,
    'wiek': 83,
    'adres': 84,
    'konto': 85,
    'jezyk': 86,
    'glowa': 87,
    'wlosy': 88,
    'zab': 89,
    'noga': 90,
    'brzuch': 91,
    'wysoki': 92,
    'niski': 93,
    'szczuply': 94,
    'chudy': 95,
    'gruby': 96,
    'piekny': 97,
    'ladny': 98,
    'brzydki': 99,
    'mlody': 100,
    'stary': 101,
    'ubrania': 102,
    'koszula': 103,
    'buty': 104,
    'skarpety': 105,
    'krawat': 106,
    'sukienka': 107,
    'sweter': 108,
    'charakter': 109,
    'mily': 110,
    'spokojny': 111,
    'cierpliwy': 112,
    'chytry': 113,
    'inteligentny': 114,
    'glupi': 115,
    'pracowity': 116,
    'leniwy': 117,
    'bogaty': 118,
    'biedny': 119,
    'uczucia': 120,
    'szczesliwy': 121,
    'smutny': 122,
    'zaskoczony': 123,
    'zly': 124,
    'zmeczony': 125,
    'glodny': 126,
    'zle sie czuje': 127,
    'bol': 128,
    'temperatura': 129,
    'jest': 130,
    'przeziebiony': 131,
    'zoladek': 132,
    'watroba': 133,
    'grypa': 134,
    'przeziebienie': 135,
    'rodzice': 136,
    'matka': 137,
    'ojciec': 138,
    'brat': 139,
    'siostra': 140,
    'dziecko': 141,
    'syn': 142,
    'corka': 143,
    'slub': 144,
    'rozwod': 145,
    'urodziny': 146,
    'boze narodzenie': 147,
    'wielkanoc': 148,
    'nowy rok': 149,
    'zaproszenie': 150,
    'klucz': 151,
    'ksiazka': 152,
    'telefon': 153,
    'aparat': 154,
    'zdjecie': 155,
    'okulary': 156,
    'talerz': 157,
    'noz': 158,
    'widelec': 159,
    'obiad':160,
    'kolacja':161,
    'lozko': 162,
    'kanapa': 163,
    'stol': 164,
    'krzeslo': 165,
    'polka': 166,
    'roslina': 167,
    'telewizor': 168,
    'kuchenka': 169,
    'lodowka': 170,
    'pralka': 177,
    'dom': 172,
    'mieszkanie': 173,
    'toaleta': 174,
    'ogrod': 175,
    'garaz': 176,
    'placic': 177,
    'reszta': 178,
    'tani': 179,
    'drogi': 180,
    'sklepy': 181,
    'piekarnia': 182,
    'supermarket': 183,
    'historyczny': 184,
    'popularny': 185,
    'interesujacy': 186,
    'nudny': 187,
    'ważny': 188,
    'wygodny': 189,
    'bezpieczny': 190,
    'niebezpieczny': 191,
    'nowy': 192,
    'stary': 193,
    'nowoczesny': 194,
    'ogromny': 195,
    'duzy': 196,
    'maly': 197,
    'wysoki': 198,
    'niski': 199,
    'waski': 200,
    'czysty': 201,
    'szeroko': 202,
    'brudny': 203,
    'miasto': 204,
    'stolica': 205,
    'centrum': 206,
    'ulica': 207,
    'park': 208,
    'parking': 209,
    'kosciol': 210,
    'kino': 211,
    'biblioteka': 212,
    'rower': 213,
    'samolot': 214,
    'autobus': 215,
    'pociag': 216,
    'wypadek samochodowy': 217,
    'metro': 218,
    'bagaz': 219
}

class DataCollector:
    def __init__(self, tmp_path):
        self.space_pressed = False
        self.right_arrow_pressed = False
        self.left_arrow_pressed = False
        self.tmp_path = tmp_path
        self.actions = np.array(list(ACTION_TO_IDX.keys()))
        self.action_idx = 0


    def on_press(self, key):
        if key == keyboard.Key.space:
            self.space_pressed = True
        elif key == keyboard.Key.right:
            self.right_arrow_pressed = True
        elif key == keyboard.Key.left:
            self.left_arrow_pressed = True


    def on_release(self, key):
        if key == keyboard.Key.space:
            self.space_pressed = False
        if key == keyboard.Key.right:
            self.right_arrow_pressed = False
        elif key == keyboard.Key.left:
            self.left_arrow_pressed = False


    def handle_action_change(self):
        if self.right_arrow_pressed:
            if self.action_idx < len(self.actions) - 1:
                self.action_idx += 1
            else:
                self.action_idx = 0
            cv2.waitKey(200)

        if self.left_arrow_pressed:
            if self.action_idx > 0:
                self.action_idx -= 1
            else:
                self.action_idx = len(self.actions) - 1
            cv2.waitKey(200)


    def annotate_sample(self, sample_num: int, action: str):
        with open(os.path.join(self.tmp_path, f'annotations.csv'), 'a') as f:
            f.write(f'{sample_num},{action}\n')


    def save_image(self, cap, sample_num, frame):
        _, image = cap.read()
        if frame is not None:
            image = cv2.resize(image, dsize=(160, 120), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(self.tmp_path, str(sample_num), f'{frame}.jpg'), image)

    def resize(self):
        for dir in os.listdir(self.tmp_path):
            sample_path = os.path.join(self.tmp_path, dir)
            
            for filename in os.listdir(sample_path):
                img = cv2.imread(os.path.join(sample_path, filename))
                cv2.imwrite(os.path.join(sample_path, filename), img)


    def record_samples(self):
        samples = [int(dir) for dir in os.listdir(self.tmp_path) if os.path.isdir(os.path.join(self.tmp_path, dir))]
        n_samples = 0 if not samples else sorted(samples)[len(samples)-1]+1
        sequences = 500  # max number of samples to record
        frames = 100  # max number of frames per sample
        cap = cv2.VideoCapture(0)

        with mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:
            action = self.actions[self.action_idx]
            listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
            listener.start()

            for sequence in range(sequences):
                while not self.space_pressed:
                    image = draw_landmarks(cap, holistic)
                    add_window_text(image, action)
                    cv2.imshow('Camera', image)

                    # Select action
                    self.handle_action_change()
                    action = self.actions[self.action_idx]

                    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
                        cap.release()
                        cv2.destroyAllWindows()
                        return

                os.makedirs(f'{os.path.join(self.tmp_path, str(n_samples))}')
                print(f"Collection data for: {action} sequence no: {sequence}.")
                countdown(cap, holistic)

                # Collect sample
                for frame in range(frames):
                    self.save_image(cap, n_samples, frame)
                    image = draw_landmarks(cap, holistic)
                    cv2.putText(image, f"Collecting frames. Action: {action} frame no: {frame}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.putText(image, f"Press ESC to stop recording", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('Camera', image)
                    if cv2.waitKey(1) == 113:  # q to stop recording
                        break
                self.annotate_sample(n_samples, action)

                if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
                    print('Finished recording. To push to remote from the default directory, run py dvc.py --command push')
                    break

                n_samples += 1

        cap.release()
        cv2.destroyAllWindows()
        listener.stop()