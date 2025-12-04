document.addEventListener('DOMContentLoaded', () => {
    const slides = document.querySelectorAll('.slide');
    const prevBtn = document.getElementById('prevBtn');
    const nextBtn = document.getElementById('nextBtn');
    const startBtn = document.getElementById('startBtn');
    const indicatorsContainer = document.getElementById('slideIndicators');

    let currentSlideIndex = 0;
    const totalSlides = slides.length;

    // Initialize Indicators
    if (indicatorsContainer) {
        slides.forEach((_, index) => {
            const dot = document.createElement('div');
            dot.classList.add('indicator-dot');
            if (index === 0) dot.classList.add('active');

            // Add click event to indicator
            dot.addEventListener('click', () => {
                goToSlide(index);
            });

            indicatorsContainer.appendChild(dot);
        });
    }

    const indicators = document.querySelectorAll('.indicator-dot');

    // Navigation Logic
    function goToSlide(index) {
        if (index < 0 || index >= totalSlides) return;

        // Update Slide Visibility
        slides.forEach((slide, i) => {
            if (i === index) {
                slide.classList.add('active');
                // Trigger animations for specific layouts
                const layout = slide.getAttribute('data-layout');
                handleLayoutAnimation(layout);
            } else {
                slide.classList.remove('active');
            }
        });

        // Update Indicators
        indicators.forEach((dot, i) => {
            dot.classList.toggle('active', i === index);
        });

        currentSlideIndex = index;
        updateButtons();
    }

    function updateButtons() {
        if (prevBtn) {
            prevBtn.disabled = currentSlideIndex === 0;
            prevBtn.style.opacity = currentSlideIndex === 0 ? '0.5' : '1';
        }
        if (nextBtn) {
            nextBtn.disabled = currentSlideIndex === totalSlides - 1;
            nextBtn.style.opacity = currentSlideIndex === totalSlides - 1 ? '0.5' : '1';
        }
    }

    function handleLayoutAnimation(layout) {
        // specific logic per layout type if needed
        console.log(`Entering layout: ${layout}`);
    }

    // Event Listeners
    if (nextBtn) nextBtn.addEventListener('click', () => goToSlide(currentSlideIndex + 1));
    if (prevBtn) prevBtn.addEventListener('click', () => goToSlide(currentSlideIndex - 1));

    if (startBtn) {
        startBtn.addEventListener('click', () => goToSlide(1));
    }

    document.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowRight' || e.key === ' ') {
            goToSlide(currentSlideIndex + 1);
        } else if (e.key === 'ArrowLeft') {
            goToSlide(currentSlideIndex - 1);
        }
    });

    // Initial State
    updateButtons();
});
