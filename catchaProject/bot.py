import asyncio

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from catchaProject.audioProcessing import AsyncAudioProcessor, Audio
from catchaProject.checkScam import ScamAnalyzer
from catchaProject.config import AUDIO_DIR, BOT_API, TRANSCRIPT_DIR

bot = Bot(token=BOT_API)
dp = Dispatcher()
processor = AsyncAudioProcessor()
sa = ScamAnalyzer()

FileLimit = 10_485_760


class States(StatesGroup):
    start = State()
    audio_wait = State()


async def send_welcome(message: Message):
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="Начать обработку аудио", callback_data="start_audio"
                )
            ]
        ]
    )
    await message.answer(
        """Привет! 👋
Я бот для распознавания мошенников по телефонным звонкам.

⚠️ Пока что я могу работать только с аудиофайлами.
Нажмите кнопку снизу чтобы начать

💡 В будущем планируется поддержка прямого распознавания звонков в реальном времени в отдельном приложении.  """,
        reply_markup=keyboard,
    )


async def start_audio_processing(call: CallbackQuery, state: FSMContext):
    await call.answer()
    await call.message.answer(
        f"Отправьте запись звонка в формате аудио (mp3, wav и т.д. не больше {FileLimit/1048576} MB), и я проанализирую её на признаки мошенничества."
    )
    await state.set_state(States.audio_wait)


async def handle_audio(message: Message, state: FSMContext):
    userID = message.from_user.id
    audio = message.audio
    audioId = audio.file_id
    if audio.file_size >= FileLimit:
        await message.answer(f"Аудиофайл должен быть не больше {FileLimit/1048576} MB")
        return
    path = AUDIO_DIR + audioId
    await bot.download(audioId, path)
    await message.answer(f"Аудиофайл '{audio.file_name}' получен. Начинаю обработку...")

    processor.queue.append(userID)
    while processor.work and processor.queue[0] != userID:
        await asyncio.sleep(5)
    processor.work = True
    audio = Audio(path)
    processor(audio)

    transcriptPath = TRANSCRIPT_DIR + audioId + ".json"
    audio.save_to_json(transcriptPath)
    processor.queue.pop(0)
    processor.work = False
    try:
        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text="Начать обработку аудио", callback_data="start_audio"
                    )
                ]
            ]
        )
        predict = sa.run(transcriptPath)
        await message.answer(f"Мошенничество: {predict[1]}%", reply_markup=keyboard)
        await state.set_state(States.start)
    except RuntimeError:
        await message.answer("Повторите попытку позже")


def register_handlers():
    dp.message.register(send_welcome, Command("start"))
    dp.callback_query.register(start_audio_processing, F.data == "start_audio")
    dp.message.register(handle_audio, States.audio_wait, F.content_type == "audio")


async def main():
    register_handlers()
    await dp.start_polling(bot, skip_updates=True)
