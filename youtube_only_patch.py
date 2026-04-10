# =========================
# YouTube-only URL filter 추가 버전
# =========================

def is_youtube_url(url: str) -> bool:
    return (
        "youtube.com" in url.lower()
        or "youtu.be" in url.lower()
    )

# =========================
# Streamlit 수정 부분
# =========================

# 기존 URL 처리 부분을 아래로 교체

url = video_url.strip()

if not is_youtube_url(url):
    st.error("❌ 유튜브 URL만 지원합니다.")
    return

try:
    with st.spinner("유튜브 영상 다운로드 중..."):
        input_path = download_video_from_url(url, td)
except Exception as e:
    st.error(str(e))
    return

# =========================
# UX 안내 추가
# =========================

st.caption("👉 유튜브 URL만 지원합니다 (youtube.com / youtu.be)")
