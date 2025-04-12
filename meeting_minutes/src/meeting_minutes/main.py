#!/usr/bin/env python
from pathlib import Path
from pydantic import BaseModel
from openai import OpenAI
from pydub.utils import make_chunks
from pydub import AudioSegment
import os
import asyncio
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from crewai.flow import Flow, listen, start
from crews.MeetingMinutes_crew.MeetingMinutes_crew import MeetingMinutesCrew

from dotenv import load_dotenv
load_dotenv()

client = OpenAI()


class MeetingMinutesState(BaseModel):
    transcript: str = ""
    minutes: str = ""

class MeetingMinutes(Flow[MeetingMinutesState]):

    def process_chunk(self, chunk_path, chunk, chunk_number):
        print(f'Transcribing chunk {chunk_number} | Beggining running_time: {datetime.now()}')
        chunk.export(chunk_path, format="wav")

        with open(chunk_path, "rb") as audio_file:  # síncrono aqui!
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
            print(f'Ending running_time: {datetime.now()}')
            return chunk_number, transcription

    @start()
    async def transcribe_meeting(self):
        print("Generating transcription")

        SCRIPT_DIR = Path(__file__).parent
        #audio_path = str(SCRIPT_DIR / "original_audio/EarningsCall.wav")
        audio_path = str(SCRIPT_DIR / "original_audio/fernando_audio.wav")
        print(f'audio_path: {audio_path}')

        # Load the audio file
        audio = AudioSegment.from_file(audio_path, format="wav")

        # Define chunk length in milliseconds (e.g., 1 minute = 60000ms)
        chunk_length_ms = 60000
        chunks = make_chunks(audio, chunk_length_ms)

        # Create chunks directory
        chunks_directory_path = 'src/meeting_minutes/audio_chunks'
        os.makedirs(chunks_directory_path, exist_ok=True)

        # Transcribe each chunk
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor() as executor:
            chunk_tasks = [
                loop.run_in_executor(
                    executor,
                    self.process_chunk,
                    f"{chunks_directory_path}/chunk_{i}.wav",
                    chunk,
                    i
                )
                for i, chunk in enumerate(chunks)
            ]
            results = await asyncio.gather(*chunk_tasks)

        chunks_transcriptions = {str(i): transcription for (i, transcription) in results}
        full_transcription = " ".join([chunks_transcriptions[str(i)].text for i in range(len(chunks_transcriptions))])
        self.state.transcript = full_transcription
        #print(f'self.state.transcript: {self.state.transcript}')

    @listen(transcribe_meeting)
    def generate_meeting_minutes(self):
        print("Generating meeting minutes")

        crew = MeetingMinutesCrew()

        inputs = {
            "transcript": self.state.transcript,
        }

        meeting_minutes = crew.crew().kickoff(inputs=inputs)

        self.state.meeting_minutes = meeting_minutes
        print(f'self.state.meeting_minutes: {self.state.meeting_minutes}')


def kickoff():
    poem_flow = MeetingMinutes()
    poem_flow.kickoff()


if __name__ == "__main__":
    kickoff()
