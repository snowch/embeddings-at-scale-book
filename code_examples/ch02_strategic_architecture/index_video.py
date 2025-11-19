# Code from Chapter 02
# Book: Embeddings at Scale

def index_video(video_path):
    """Index video with multiple modalities"""
    # Extract frames (visual)
    frames = extract_key_frames(video_path, num_frames=10)
    frame_embeddings = [encoder.encode_image(frame) for frame in frames]
    video_visual_emb = torch.stack(frame_embeddings).mean(dim=0)

    # Extract audio
    audio = extract_audio(video_path)
    audio_emb = encoder.encode_audio(audio)

    # Extract and embed transcription
    transcription = speech_to_text(audio)
    text_emb = encoder.encode_text(transcription)

    # Fused multi-modal video embedding
    video_emb = ModalityFusion.early_fusion(
        [video_visual_emb, audio_emb, text_emb],
        weights=[0.5, 0.2, 0.3]
    )

    return video_emb
