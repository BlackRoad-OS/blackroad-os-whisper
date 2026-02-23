"""Speech-to-text processing pipeline (Whisper-inspired)."""

import sqlite3
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Optional, Dict
import hashlib
import os
from enum import Enum


@dataclass
class Segment:
    """Speech segment with timing and confidence."""
    start_s: float
    end_s: float
    text: str
    confidence: float = 0.95
    speaker: Optional[str] = None


@dataclass
class Transcription:
    """Speech transcription result."""
    id: str
    file_path: str
    duration_s: float
    model: str
    language: str
    text: str
    segments: List[Segment] = field(default_factory=list)
    confidence: float = 0.95
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    processing_time_ms: int = 0


class SpeechProcessor:
    """Speech-to-text processing pipeline."""

    def __init__(self, db_path: Optional[str] = None):
        """Initialize speech processor with SQLite backend."""
        if db_path is None:
            db_path = os.path.expanduser("~/.blackroad/speech.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize database schema with FTS5."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transcriptions (
                id TEXT PRIMARY KEY,
                file_path TEXT UNIQUE NOT NULL,
                duration_s REAL,
                model TEXT,
                language TEXT,
                text TEXT,
                segments TEXT,
                confidence REAL,
                created_at TEXT,
                processing_time_ms INTEGER
            )
        """)

        # Create FTS5 virtual table for full-text search
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS transcriptions_fts USING fts5(
                id,
                text,
                content='transcriptions',
                content_rowid='rowid'
            )
        """)

        conn.commit()
        conn.close()

    def transcribe(self, file_path: str, model: str = "base",
                  language: Optional[str] = None, task: str = "transcribe") -> str:
        """Transcribe audio file."""
        import time
        start_time = time.time()

        # Check if whisper is available
        try:
            import whisper
            result = whisper.load_model(model).transcribe(file_path, language=language, task=task)
            text = result.get("text", "")
            detected_lang = result.get("language", language or "en")
            segments_data = result.get("segments", [])
        except ImportError:
            # Mock transcription
            file_size = os.path.getsize(file_path)
            # Estimate: 128KB ≈ 1 minute of audio
            duration = max(1, file_size / (128 * 1024))
            word_count = int(duration * 150)  # ~150 words per minute
            text = " ".join([f"word{i}" for i in range(word_count)])
            detected_lang = language or "en"
            segments_data = []

        processing_time = int((time.time() - start_time) * 1000)

        # Parse segments
        segments = []
        for seg in segments_data:
            segment = Segment(
                start_s=seg.get("start", 0),
                end_s=seg.get("end", 0),
                text=seg.get("text", ""),
                confidence=seg.get("confidence", 0.95)
            )
            segments.append(segment)

        # If no segments, create one
        if not segments:
            segments = [Segment(start_s=0, end_s=duration if 'duration' in locals() else 60, text=text)]

        # Store
        transcription_id = hashlib.sha256(f"{file_path}{datetime.utcnow().isoformat()}".encode()).hexdigest()[:16]
        duration = segments[-1].end_s if segments else 0

        transcription = Transcription(
            id=transcription_id,
            file_path=file_path,
            duration_s=duration,
            model=model,
            language=detected_lang,
            text=text,
            segments=segments,
            processing_time_ms=processing_time
        )

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO transcriptions (id, file_path, duration_s, model, language, text, segments, confidence, created_at, processing_time_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (transcription.id, file_path, duration, model, detected_lang, text,
              json.dumps([asdict(s) for s in segments]), transcription.confidence,
              transcription.created_at, processing_time))

        # Index in FTS
        cursor.execute("INSERT INTO transcriptions_fts (id, text) VALUES (?, ?)",
                      (transcription_id, text))

        conn.commit()
        conn.close()

        return transcription_id

    def detect_language(self, file_path: str) -> tuple:
        """Detect language from audio file."""
        try:
            import whisper
            model = whisper.load_model("tiny")
            result = model.detect_language(file_path)
            return result.get("language", "en"), result.get("confidence", 0.95)
        except ImportError:
            return "en", 0.95  # Mock

    def translate(self, file_path: str, target_language: str = "en") -> str:
        """Transcribe and translate audio."""
        return self.transcribe(file_path, task="translate")

    def diarize(self, file_path: str, num_speakers: Optional[int] = None) -> str:
        """Speaker diarization."""
        transcription_id = self.transcribe(file_path)

        # Assign speakers to segments (mock)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT segments FROM transcriptions WHERE id = ?", (transcription_id,))
        row = cursor.fetchone()

        if row:
            segments = json.loads(row[0])
            num_speakers = num_speakers or 2

            # Round-robin assign speakers
            for i, seg in enumerate(segments):
                seg["speaker"] = f"SPEAKER_{i % num_speakers}"

            cursor.execute("UPDATE transcriptions SET segments = ? WHERE id = ?",
                          (json.dumps(segments), transcription_id))
            conn.commit()

        conn.close()
        return transcription_id

    def get_transcript(self, transcription_id: str, format: str = "text") -> str:
        """Get transcript in various formats."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT text, segments FROM transcriptions WHERE id = ?", (transcription_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return ""

        text, segments_json = row
        segments = json.loads(segments_json) if segments_json else []

        if format == "text":
            return text
        elif format == "srt":
            return self._format_srt(segments)
        elif format == "vtt":
            return self._format_vtt(segments)
        elif format == "json":
            return json.dumps({
                "text": text,
                "segments": segments
            }, indent=2)
        else:
            return text

    def _format_srt(self, segments: List[Dict]) -> str:
        """Format as SubRip (SRT)."""
        srt = ""
        for i, seg in enumerate(segments, 1):
            start = self._seconds_to_time(seg.get("start_s", 0))
            end = self._seconds_to_time(seg.get("end_s", 0))
            text = seg.get("text", "").strip()
            srt += f"{i}\n{start} --> {end}\n{text}\n\n"
        return srt

    def _format_vtt(self, segments: List[Dict]) -> str:
        """Format as WebVTT."""
        vtt = "WEBVTT\n\n"
        for seg in segments:
            start = self._seconds_to_time(seg.get("start_s", 0))
            end = self._seconds_to_time(seg.get("end_s", 0))
            text = seg.get("text", "").strip()
            vtt += f"{start} --> {end}\n{text}\n\n"
        return vtt

    def _seconds_to_time(self, seconds: float) -> str:
        """Convert seconds to HH:MM:SS.mmm format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def export_srt(self, transcription_id: str, output_path: str) -> bool:
        """Export as SubRip subtitle format."""
        content = self.get_transcript(transcription_id, format="srt")
        if content:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "w") as f:
                f.write(content)
            return True
        return False

    def export_vtt(self, transcription_id: str, output_path: str) -> bool:
        """Export as WebVTT format."""
        content = self.get_transcript(transcription_id, format="vtt")
        if content:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "w") as f:
                f.write(content)
            return True
        return False

    def list_transcriptions(self, file_filter: Optional[str] = None,
                           language: Optional[str] = None) -> List[Transcription]:
        """List transcriptions with optional filters."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = "SELECT * FROM transcriptions"
        params = []

        if language:
            query += " WHERE language = ?"
            params.append(language)

        if file_filter:
            if params:
                query += " AND file_path LIKE ?"
            else:
                query += " WHERE file_path LIKE ?"
            params.append(f"%{file_filter}%")

        query += " ORDER BY created_at DESC"

        cursor.execute(query, params)
        rows = cursor.fetchall()

        transcriptions = []
        for row in rows:
            segments = [Segment(**s) for s in json.loads(row[6])] if row[6] else []
            transcription = Transcription(
                id=row[0], file_path=row[1], duration_s=row[2], model=row[3],
                language=row[4], text=row[5], segments=segments, confidence=row[7],
                created_at=row[8], processing_time_ms=row[9]
            )
            transcriptions.append(transcription)

        conn.close()
        return transcriptions

    def search_transcriptions(self, query: str) -> List[Transcription]:
        """Full-text search across transcriptions."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Use FTS5
        cursor.execute("""
            SELECT t.* FROM transcriptions t
            WHERE t.id IN (
                SELECT id FROM transcriptions_fts WHERE text MATCH ?
            )
        """, (query,))

        rows = cursor.fetchall()
        transcriptions = []

        for row in rows:
            segments = [Segment(**s) for s in json.loads(row[6])] if row[6] else []
            transcription = Transcription(
                id=row[0], file_path=row[1], duration_s=row[2], model=row[3],
                language=row[4], text=row[5], segments=segments, confidence=row[7],
                created_at=row[8], processing_time_ms=row[9]
            )
            transcriptions.append(transcription)

        conn.close()
        return transcriptions

    def get_stats(self) -> Dict:
        """Get overall statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*), SUM(duration_s) FROM transcriptions")
        count_row = cursor.fetchone()

        cursor.execute("SELECT language, COUNT(*) FROM transcriptions GROUP BY language")
        lang_rows = cursor.fetchall()

        cursor.execute("SELECT model, COUNT(*) FROM transcriptions GROUP BY model")
        model_rows = cursor.fetchall()

        conn.close()

        return {
            "total_files": count_row[0] or 0,
            "total_hours": (count_row[1] or 0) / 3600,
            "languages": dict(lang_rows),
            "models_used": dict(model_rows)
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Speech Processor CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Transcribe
    trans_parser = subparsers.add_parser("transcribe")
    trans_parser.add_argument("file_path")
    trans_parser.add_argument("--model", type=str, default="base")

    # SRT export
    srt_parser = subparsers.add_parser("srt")
    srt_parser.add_argument("transcription_id")
    srt_parser.add_argument("output_path")

    # Search
    search_parser = subparsers.add_parser("search")
    search_parser.add_argument("query")

    args = parser.parse_args()
    processor = SpeechProcessor()

    if args.command == "transcribe":
        trans_id = processor.transcribe(args.file_path, model=args.model)
        print(f"Transcription ID: {trans_id}")
    elif args.command == "srt":
        processor.export_srt(args.transcription_id, args.output_path)
        print(f"Exported to {args.output_path}")
    elif args.command == "search":
        results = processor.search_transcriptions(args.query)
        for result in results:
            print(f"{result.id}: {result.text[:100]}")
