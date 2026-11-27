# Модели для конвейера синтеза (лабораторная 5)

Веса в git не хранятся: GitHub не принимает LFS-файлы в публичных форках, а модель
лабы 4 больше лимита в 100 МБ. Они распространяются архивом `tts_models_lab5.zip` (~210 МБ).

## Что в архиве

Пути внутри архива совпадают с путями в репозитории:

| файл | размер | что это |
|---|---|---|
| `labs/lab5_service/models/acoustic_pitch_energy/best_model_31650.pth` | 153 МБ | FastSpeech 2 из лабы 4, вариант `pitch_energy`, только веса (без оптимизатора) |
| `labs/lab5_service/models/acoustic_pitch_energy/config.json` | 9 КБ | конфиг coqui к ней |
| `labs/lab5_service/models/vocoder_gta/generator.ckpt` | 53 МБ | HiFi-GAN, дообученный на GTA-мелах RUSLAN (отчёт, раздел 9) |
| `labs/lab5_service/models/vocoder_gta/hyperparams.yaml` | 4 КБ | гиперпараметры SpeechBrain к нему |
| `labs/lab3_g2p/data/russian_mfa.dict` | 24 МБ | словарь MFA для транскриптора лабы 3 |
| `labs/lab3_g2p/data/v3/lexicon.csv` | 2 МБ | лексикон RUSLAN для транскриптора лабы 3 |
| `labs/lab5_service/MODELS.md` | - | этот файл |

Остальные модели в репозитории уже есть: `labs/lab2_pauses/models/pause_model.pkl`,
`labs/lab3_g2p/models/oov_model.pkl.gz`, `labs/lab3_g2p/models/variant_chooser.pkl.gz`.
Исходный HiFi-GAN (`speechbrain/tts-hifigan-libritts-22050Hz`) скачивается с Hugging Face сам
при первом запуске.

## Куда положить

Распаковать в **корень репозитория** - файлы встанут на свои места:

```powershell
Expand-Archive -Path tts_models_lab5.zip -DestinationPath D:\path\to\tts_labs_itmo -Force
```

Проверка:

```powershell
cd labs\lab5_service
python -c "from tts_pipeline import TTSPipeline; w, t = TTSPipeline()('Привет, мир!'); print(len(w))"
```

## Запуск

```powershell
cd labs\lab5_service
python service.py                  # исходный вокодер LibriTTS
python service.py --vocoder gta    # дообученный на GTA-мелах (им синтезирован набор для MOS)
```

Демо - `http://127.0.0.1:8000/demo`, REST - `POST /synthesize` (подробнее в отчёте, раздел 10).
Модель лабы 4 по умолчанию берётся из `models/acoustic_pitch_energy`; другой прогон -
`--run <папка с config.json и best_model_*.pth>`.
