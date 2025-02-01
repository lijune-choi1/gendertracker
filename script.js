document.addEventListener('DOMContentLoaded', () => {
    // Modal functionality
    const helpButton = document.querySelector('.help-button');
    const modalOverlay = document.querySelector('.modal-overlay');
    const modalClose = document.querySelector('.modal-close');

    helpButton.addEventListener('click', () => {
        modalOverlay.style.display = 'flex';
    });

    modalClose.addEventListener('click', () => {
        modalOverlay.style.display = 'none';
    });

    modalOverlay.addEventListener('click', (e) => {
        if (e.target === modalOverlay) {
            modalOverlay.style.display = 'none';
        }
    });
    const inputField = document.querySelector('.input-field');
    
    inputField.addEventListener('input', () => {
        // Here you could add logic to analyze the text and update suggestions
    });

    const acceptButtons = document.querySelectorAll('.btn-accept');
    const dismissButtons = document.querySelectorAll('.btn-dismiss');

    acceptButtons.forEach(button => {
        button.addEventListener('click', () => {
            const suggestionItem = button.closest('.suggestions-section');
            suggestionItem.style.display = 'none';
        });
    });

    dismissButtons.forEach(button => {
        button.addEventListener('click', () => {
            const suggestionItem = button.closest('.suggestions-section');
            suggestionItem.style.display = 'none';
        });
    });
});