document.addEventListener("DOMContentLoaded", () => {
    const features = document.querySelectorAll(".feature");

    features.forEach((feature, index) => {
        setTimeout(() => {
            feature.style.opacity = "1";
            feature.style.transform = "translateY(0)";
        }, index * 300);
    });
    
});
