# bot/bot.py
import sys
from pathlib import Path
import os

# --- ИСПРАВЛЕНИЕ ИМПОРТА ---
# Добавляем корневую директорию проекта в Python path
sys.path.insert(0, str(Path(file).resolve().parent.parent))

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    MessageHandler,
    CommandHandler,
    ContextTypes,
    filters,
)

from config import BOT_TOKEN
from Classifier.predict import predict_all

TMP = "bot_tmp"
os.makedirs(TMP, exist_ok=True)

# Перевод классов на русский
CLASS_RU = {
    "WithMask": "В маске 😷",
    "WithoutMask": "Без маски 😶",
}

# Русские названия моделей
MODEL_RU = {
    "HOG+SVM": "Классика (контуры) — HOG + SVM",
    "LBP+LR":  "Классика (текстуры) — LBP + LogisticRegression",
    "CNN":     "Нейросеть — MobileNetV2",
}

ORDER = ["HOG+SVM", "LBP+LR", "CNN"]


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! 👋\n"
        "Пришли фото лица — я проверю, есть ли маска, и покажу результат трёх моделей."
    )


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Как пользоваться:\n"
        "1) Отправь фото (обычным изображением).\n"
        "2) Я отвечу результатами трёх моделей и укажу, какая из них была самой уверенной на этом фото."
    )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo = update.message.photo[-1]
    file = await context.bot.get_file(photo.file_id)

    path = os.path.join(TMP, f"{photo.file_id}.jpg")
    await file.download_to_drive(path)

    try:
        # {"HOG+SVM": ("WithMask", 0.99), ...}
        res = predict_all(path)

        # Лучшая модель на этом изображении — по максимальной уверенности
        best_model, (best_cls, best_conf) = max(res.items(), key=lambda kv: kv[1][1])

        lines = []
        lines.append("🧠 *Результаты распознавания маски:*")
        lines.append("")

        for key in ORDER:
            cls, conf = res[key]
            cls_text = CLASS_RU.get(cls, cls)
            conf_pct = conf * 100.0

            lines.append(f"• *{MODEL_RU.get(key, key)}*")
            lines.append(f"  → *{cls_text}*  _(уверенность: {conf_pct:.2f}%)_")
            lines.append("")

        best_cls_text = CLASS_RU.get(best_cls, best_cls)
        lines.append("🏆 *Самая уверенная модель на этом фото:*")
        lines.append(f"*{MODEL_RU.get(best_model, best_model)}* → *{best_cls_text}* ({best_conf*100:.2f}%)")

        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

    except Exception as e:
        await update.message.reply_text(f"Ошибка при обработке изображения: {e}")

    finally:
        if os.path.exists(path):
            os.remove(path)


async def handle_non_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Пожалуйста, пришли именно фото 🙂")


def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))

    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(~filters.PHOTO & filters.ALL, handle_non_photo))

    app.run_polling()


if name == "main":
    main()


