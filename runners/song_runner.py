"""
High-level entry point for the NAO jazz song generation pipeline.

This module wires a NAO robot and optional logger into the
`generate_jazz_song_with_squats` routine from `modules.jazz_song_generator`.

It is synchronous: it will return only after the song has been generated
(or failed) and playback has finished.
"""

from __future__ import annotations

from modules.song_generator import song_generation_with_exercise


def run_song(
    nao,
    dialogflow_cx,
    session_id: int,
    logger=None,
    max_wait_time: float = 600.0,
    nao_ip: str | None = None,
) -> None:
    """Run the synchronous song generation pipeline on a NAO robot.

    This is a convenience wrapper around `song_generation_with_exercise` that:
      - Forwards the NAO handle and optional logger.
      - Asks the user for the desired genre via Dialogflow.
      - Waits for Suno to generate a track.
      - Lets NAO do "exercise" while waiting.
      - Plays the generated song once ready.
    """
    _ = song_generation_with_exercise(
        nao=nao,
        dialogflow_cx=dialogflow_cx,
        session_id=session_id,
        logger=logger,
        max_wait_time=max_wait_time,
    )