"""
High-level entry point for the NAO jazz song generation pipeline.

This module wires a NAO robot and optional logger into the
`generate_jazz_song_with_squats` routine from `modules.jazz_song_generator`.

It is synchronous: it will return only after the song has been generated
(or failed) and playback has finished.
"""

from __future__ import annotations

from modules.song_generator import generate_jazz_song_with_squats


def run_jazz_song(
    nao,
    nao_ip: str,
    logger=None,
    max_wait_time: float = 600.0,
) -> None:
    """Run the synchronous jazz song generation pipeline on a NAO robot.

    This is a convenience wrapper around `generate_jazz_song_with_squats` that:
      - Forwards the NAO handle and optional logger.
      - Waits for Suno to generate a jazz track.
      - Lets NAO do "squats" while waiting.
      - Plays the generated song once ready.

    Args:
        nao: NAO device handle with `.motion`, `.autonomous`, and `.speaker`.
        nao_ip: IP address of the NAO robot (kept for API symmetry, not used).
        logger: Optional logger-like object.
        max_wait_time: Maximum time in seconds to wait for Suno's generation
            before giving up.
    """
    # Currently we don't need nao_ip in the generator, but we keep it in the
    # signature so this runner matches the pattern of other runners.
    _ = generate_jazz_song_with_squats(
        nao=nao,
        logger=logger,
        max_wait_time=max_wait_time,
    )