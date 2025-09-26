import asyncio
import logging

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

from catchaProject.audioProcessing import AudioProcessor, Audio
from catchaProject.checkScam import ScamAnalyzer
from catchaProject.config import AUDIO_DIR, BOT_API, TRANSCRIPT_DIR

bot = Bot(token=BOT_API)
dp = Dispatcher()
processor = AudioProcessor()
sa = ScamAnalyzer()

semaphore = asyncio.Semaphore()
queue = []

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
    userId = message.from_user.id
    audio = message.audio
    audioId = audio.file_id
    if audio.file_size >= FileLimit:
        await message.answer(f"Аудиофайл должен быть не больше {FileLimit/1048576} MB")
        return

    queue.append(userId)
    pos = len(queue)
    if pos > 1:
        await message.answer(f"вы {pos}-й в очереди")
    async with semaphore:
        path = AUDIO_DIR + audioId
        await bot.download(audioId, path)
        await message.answer(f"Аудиофайл '{audio.file_name}' получен. Начинаю обработку...")
        logging.info(f"Файл {audioId} от {userId} {message.from_user.username}")

        audio = Audio(path)
        processor(audio)

        transcriptPath = TRANSCRIPT_DIR + audioId + ".json"
        audio.save_to_json(transcriptPath)
        try:
            keyboard = InlineKeyboardMarkup(
                inline_keyboard=[
                    [
                        InlineKeyboardButton(
                            text="Обработать еще", callback_data="start_audio"
                        )
                    ]
                ]
            )
            predict = sa.run(transcriptPath)
            answer = f"Мошенничество: {predict[1]}%"
            logging.info(audioId + " " + answer)
            await message.answer(answer, reply_markup=keyboard)
            await state.set_state(States.start)
        except RuntimeError:
            await message.answer("Повторите попытку позже")

        queue.pop(0)


def register_handlers():
    dp.message.register(send_welcome, Command("start"))
    dp.callback_query.register(start_audio_processing, F.data == "start_audio")
    dp.message.register(handle_audio, States.audio_wait, F.content_type == "audio")


async def main():
    register_handlers()
    await dp.start_polling(bot, skip_updates=True)
