import os
import cv2
import time
import ollama
import pytesseract
import numpy as np
from dotenv import load_dotenv
from datetime import datetime, timedelta
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    CallbackQueryHandler,
    CallbackContext,
)

# If OS is Windows, set pytesseract path
if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Hardcoded test time (change this for different test times)
TEST_DINNER_TIME = "17:52"  

async def start(update: Update, context: CallbackContext):
    """Sets up the bot with a /start command."""
    now = datetime.now().strftime("%d-%m-%Y, %H:%M:%S")
    await update.message.reply_text(
        f"👋 Hello! I'm your medication assistant bot. The time now is {now}.\n\n"
        "Please take a vertical, clear, well-lit picture of your medication. \n\n"
        "I will remind you to take it at the appropriate time."
    )

async def extract_text_from_meds(update: Update, context: CallbackContext):
    """Extracts medicine names from prescription images using OCR and asks for confirmation."""
    if update.message.photo:
        photo_file = await update.message.photo[-1].get_file()
    elif update.message.document and update.message.document.mime_type.startswith("image/"):
        photo_file = await update.message.document.get_file()
    else:
        await update.message.reply_text("⚠️ Please send a **photo or an image file**.")
        return

    image_path = "meds_image.jpg"
    await photo_file.download_to_drive(image_path)

    img = cv2.imread(image_path)
    if img is None:
        await update.message.reply_text("⚠️ Error: Could not load image.")
        return

    # Convert to grayscale and apply OCR
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, processed_img = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    extracted_text = pytesseract.image_to_string(processed_img, config="--psm 6")

    if extracted_text.strip():
        context.user_data["extracted_meds"] = extracted_text.strip()

        keyboard = [[InlineKeyboardButton("✅ Confirm", callback_data="confirm_med"),
                     InlineKeyboardButton("❌ Retry", callback_data="retry_med")]]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(
            f"📄 **Extracted Medication Name(s):**\n```{extracted_text.strip()}```\n\nDoes this look correct?",
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )
    else:
        await update.message.reply_text("⚠️ No readable text found. Try sending a clearer image.")

async def button_handler(update: Update, context: CallbackContext):
    """Handles confirmation or retrying of the extracted medicine name."""
    query = update.callback_query
    await query.answer()  # Acknowledge button click

    if query.data == "confirm_med":
        extracted_meds = context.user_data.get("extracted_meds", "Unknown")
        await query.message.edit_text(f"✅ Confirmed! Please wait a few moments...\n```\n{extracted_meds}```", parse_mode="Markdown")

        # Fetch medical advice
        await get_info_from_llm(update, context, extracted_meds)

        # Hardcoded reminder scheduling (Dinner at 6:00 PM)
        await schedule_hardcoded_reminder(update, context, extracted_meds)

    elif query.data == "retry_med":
        await query.message.edit_text("🔄 Please send another image.")

async def get_info_from_llm(update: Update, context: CallbackContext, extracted_text):
    """Fetches medication details (NO scheduling here)."""
    chat_id = update.effective_chat.id

    prompt = f"""
    You are a helpful medical assistant. The user has asked for information
    about the following medicine and dosages from their prescription label: {extracted_text}.

    Your task:
    - Identify **each medicine name** mentioned.
    - Provide **a brief explanation** of what the medicine is used for.
    - Include **dosage recommendations** if relevant
    - List **important precautions** in bullet points.
    - Mention **common side effects**.

    Keep the response **short, concise, and easy to read in a Telegram message**.
    """

    try:
        response = ollama.chat(model="qwen2.5:7b", messages=[{"role": "user", "content": prompt}])
        generated_text_obj = response["message"]
        generated_text = generated_text_obj["content"]

        await context.bot.send_message(chat_id, text=f"💊 **Medication Information:**\n\n{generated_text}", parse_mode="Markdown")

    except Exception as e:
        print(f"❌ Qwen 2.5 Error: {e}")
        await context.bot.send_message(chat_id=chat_id, text="❌ Error retrieving medication details.")

async def schedule_hardcoded_reminder(update: Update, context: CallbackContext, extracted_meds):
    """Schedules a hardcoded reminder for 5:51 PM (Dinner time)."""
    chat_id = update.effective_chat.id

    # Convert string time to a datetime object
    now = datetime.now()
    reminder_time = datetime.strptime(TEST_DINNER_TIME, "%H:%M").replace(year=now.year, month=now.month, day=now.day)

    # If time has already passed today, schedule for tomorrow
    if reminder_time < now:
        reminder_time += timedelta(days=1)

    # Calculate delay in seconds
    delay = (reminder_time - now).total_seconds()

    # Schedule the reminder
    job_queue = context.application.job_queue
    job_queue.run_once(send_medication_reminder, delay, chat_id=chat_id, name=f"{extracted_meds} at {TEST_DINNER_TIME}")

    await context.bot.send_message(chat_id, text=f"✅ **Reminder set for {TEST_DINNER_TIME}**.")

async def send_medication_reminder(context: CallbackContext):
    """Sends a reminder message at the scheduled time."""
    job = context.job
    await context.bot.send_message(job.chat_id, text=f"💊 **Reminder:** Time to take your medication!\n\n{job.name}")

def main():
    """Start the bot using polling."""
    app = Application.builder().token(TOKEN).build()

    # Start JobQueue
    job_queue = app.job_queue
    job_queue.start()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, extract_text_from_meds))
    app.add_handler(CallbackQueryHandler(button_handler))

    print("✅ Bot is running... Listening for messages")
    app.run_polling()

if __name__ == "__main__":
    main()
