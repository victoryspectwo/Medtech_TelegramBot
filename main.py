import os
import schedule
import cv2
#import asyncio
import time
import ollama
import pytesseract
#import psycopg2
import openai
import numpy as np
from dotenv import load_dotenv
from datetime import datetime, timedelta
from langchain_ollama import OllamaLLM
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackQueryHandler, CallbackContext



load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
DB_URL = os.getenv("DATABASE_URL") #db not utilised yet
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") #temporarily using OpenAI key cause GPU usage is killing my laptop

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


async def start(update: Update, context: CallbackContext): # set up /start command
    now = datetime.now().strftime("%d-%m-%Y, %H:%M:%S")
    await update.message.reply_text(
        f"👋 Hello! I'm your medication assistant bot. The time now is {now}.\n\n"
        "Please take a vertical, clear, well lit picture of your medication. \n\n"
        "I will remind you when to take them throughout the day."
    )


async def extract_text_from_meds(update: Update, context: CallbackContext):
    """Extracts medicine names from prescription images using OCR and asks for confirmation."""
    
    # Check if image is sent as a photo or a document
    if update.message.photo:
        photo_file = await update.message.photo[-1].get_file()
    elif update.message.document and update.message.document.mime_type.startswith("image/"):
        photo_file = await update.message.document.get_file()
    else:
        await update.message.reply_text("⚠️ Please send a **photo or an image file**.")
        return

    # Save the image
    image_path = "meds_image.jpg"
    await photo_file.download_to_drive(image_path)

    # Read image using OpenCV
    img = cv2.imread(image_path)

    if img is None:
        await update.message.reply_text("⚠️ Error: Could not load image.")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Binary Thresholding
    _, processed_img = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # Extract text using OCR
    extracted_text = pytesseract.image_to_string(processed_img, config="--psm 6")  

    # Send the extracted text back to the user for confirmation
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
        await query.message.edit_text(f"✅ Confirmed! Fetching information for:\n```{extracted_meds}```", parse_mode="Markdown")

        # Fetch medical advice from Llama 3.2
        await get_info_from_llm(update, context, extracted_meds)

    elif query.data == "retry_med":
        await query.message.edit_text("🔄 Please send another image.")


# Initialize OpenAI client correctly
client = openai.OpenAI(api_key=OPENAI_API_KEY)


async def get_info_from_llm(update: Update, context: CallbackContext, extracted_text):
    if not extracted_text.strip():
        await context.bot.send_message(chat_id=update.effective_chat.id, text="⚠️ Unable to retrieve the medication name.")
        return
    
    print(f"Sending medicine to GPT-4: {extracted_text}") 

    prompt = f"""
    You are a helpful medical assistant. The user has asked for information
    about the following medicine and dosages from their prescription label: {extracted_text}.

    Your task:
    - Identify **each medicine name** mentioned.
    - Provide **a brief explanation** of what the medicine is used for.
    - Include **dosage recommendations** if relevant.
    - List **important precautions** in bullet points (e.g., "Do not take on an empty stomach").
    - Mention **common side effects**.

    ---
    ### **Format your response like this:**
    💊 **Medication Name**  
   - **Uses:** Short and clear description.  
   - **Dosage:** Clearly mention when and how to take it.  
   - **Precautions:** 🔴 (Red emojis for warnings) List important things to avoid.  
   - **Side Effects:** ⚠️ (Yellow emoji for mild side effects) Short and simple list.

    Remind the user when to take it at the time of day. Keep the response **short, concise, and easy to read in a Telegram message.**.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4",  # Use "gpt-3.5-turbo" if needed
            messages=[{"role": "user", "content": prompt}],
        )

        generated_text = response.choices[0].message.content.strip()

        await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=f"💊 **Medication Instructions:**\n\n{generated_text}",
                parse_mode="Markdown"
            )

    except Exception as e:
        print(f"❌ OpenAI GPT-4 Error: {e}")
        await context.bot.send_message(chat_id=update.effective_chat.id, text="❌ Error retrieving medical advice.")


#old ollama try-except block (for use if needed)
    """
    try:
        response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])
        generated_text = response["message"]

        # Send response to Telegram using chat_id (since update.message may be None)
        await context.bot.send_message(chat_id=update.effective_chat.id, text=f"💊 **Medication Instructions:**\n\n{generated_text}")

    except Exception as e:
        print(f"❌ Llama 3.2 Error: {e}")  # Debugging
        await context.bot.send_message(chat_id=update.effective_chat.id, text="❌ Error retrieving medical advice.")
    """


def main():
    """ Start the bot using polling """
    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))  # Handles /start command
    app.add_handler(MessageHandler(filters.PHOTO, extract_text_from_meds))  
    app.add_handler(CallbackQueryHandler(button_handler))  # Handles confirmation buttons

    print("✅ Bot is running... Listening for messages")
    app.run_polling()

if __name__ == "__main__":
    main()

