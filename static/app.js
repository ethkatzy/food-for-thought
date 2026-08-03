(function () {
    const form = document.getElementById('user-form');
    const submitBtn = document.getElementById('submit-btn');
    const userIdInput = document.getElementById('user_id');
    const statusEl = document.getElementById('status');
    const legendEl = document.getElementById('legend');
    const resultsEl = document.getElementById('results');
    const cardTemplate = document.getElementById('card-template');
    const ratedSectionEl = document.getElementById('rated-section');
    const ratedSubtitleEl = document.getElementById('rated-subtitle');
    const ratedTableBodyEl = document.getElementById('rated-table-body');

    const COLD_START_DELAY_MS = 4000;

    function setStatus(message, kind) {
        if (!message) {
            statusEl.hidden = true;
            statusEl.textContent = '';
            statusEl.className = 'status';
            return;
        }
        statusEl.hidden = false;
        statusEl.className = 'status' + (kind ? ' ' + kind : '');
        statusEl.innerHTML = '';
        if (kind === 'loading') {
            const spinner = document.createElement('span');
            spinner.className = 'spinner';
            statusEl.appendChild(spinner);
        }
        statusEl.appendChild(document.createTextNode(message));
    }

    function clearResults() {
        resultsEl.hidden = true;
        resultsEl.innerHTML = '';
        legendEl.hidden = true;
        ratedSectionEl.hidden = true;
        ratedTableBodyEl.innerHTML = '';
    }

    function iconString(filledChar, emptyChar, score) {
        const filled = Math.max(0, Math.min(5, Math.round(score)));
        return filledChar.repeat(filled) + emptyChar.repeat(5 - filled);
    }

    function renderDots(container, score) {
        container.innerHTML = '';
        const filled = Math.max(0, Math.min(5, Math.round(score)));
        for (let i = 0; i < 5; i++) {
            const dot = document.createElement('span');
            dot.className = 'dot ' + (i < filled ? 'dot-filled' : 'dot-empty');
            container.appendChild(dot);
        }
    }

    function renderResults(recommendations) {
        clearResults();
        if (!recommendations.length) {
            setStatus('No recommendations found for that user yet.', 'empty');
            return;
        }

        recommendations.forEach((rec, index) => {
            const node = cardTemplate.content.cloneNode(true);

            const img = node.querySelector('.card-image');
            const fallback = node.querySelector('.card-image-fallback');
            if (rec.image) {
                img.src = rec.image;
                img.alt = rec.name;
                img.addEventListener('error', () => {
                    img.hidden = true;
                    fallback.hidden = false;
                });
            } else {
                img.hidden = true;
                fallback.hidden = false;
            }

            node.querySelector('.card-rank').textContent = '#' + (index + 1);
            node.querySelector('.card-title').textContent = rec.name;

            const matchMetric = node.querySelector('.match-metric');
            renderDots(matchMetric.querySelector('.metric-icons'), rec.score);
            matchMetric.querySelector('.metric-text').textContent = rec.score.toFixed(1) + ' / 5';

            const foodComMetric = node.querySelector('.foodcom-metric');
            if (rec.foodComRating != null) {
                foodComMetric.querySelector('.metric-icons').textContent = iconString('★', '☆', rec.foodComRating);
                foodComMetric.querySelector('.metric-text').textContent =
                    rec.foodComRating.toFixed(1) + ' / 5 (' + rec.foodComReviewCount + ')';
            } else {
                foodComMetric.classList.add('no-rating');
                foodComMetric.querySelector('.metric-icons').textContent = iconString('☆', '☆', 0);
                foodComMetric.querySelector('.metric-text').textContent = 'No ratings yet';
            }

            const link = node.querySelector('.card-link');
            link.href = rec.url;

            resultsEl.appendChild(node);
        });

        resultsEl.hidden = false;
        legendEl.hidden = false;
    }

    function renderRatedRecipes(ratedRecipes, ratedRecipesTotal) {
        if (!ratedRecipes.length) {
            return;
        }

        ratedSubtitleEl.textContent = ratedRecipesTotal > ratedRecipes.length
            ? `Showing the ${ratedRecipes.length} highest-rated of ${ratedRecipesTotal} recipes this user has rated — these are what shaped the recommendations above.`
            : `All ${ratedRecipesTotal} recipe${ratedRecipesTotal === 1 ? '' : 's'} this user has rated — these are what shaped the recommendations above.`;

        ratedRecipes.forEach((r) => {
            const row = document.createElement('tr');

            const nameCell = document.createElement('td');
            nameCell.className = 'rated-name';
            const link = document.createElement('a');
            link.href = r.url;
            link.target = '_blank';
            link.rel = 'noopener noreferrer';
            link.textContent = r.name;
            nameCell.appendChild(link);

            const ratingCell = document.createElement('td');
            ratingCell.className = 'rated-rating';
            ratingCell.textContent = iconString('★', '☆', r.rating);

            row.appendChild(nameCell);
            row.appendChild(ratingCell);
            ratedTableBodyEl.appendChild(row);
        });

        ratedSectionEl.hidden = false;
    }

    form.addEventListener('submit', async (event) => {
        event.preventDefault();
        clearResults();
        submitBtn.disabled = true;
        setStatus('Finding your recommendations…', 'loading');

        const coldStartTimer = setTimeout(() => {
            setStatus('Still working — the server may be waking up from being idle, this can take up to a minute.', 'loading');
        }, COLD_START_DELAY_MS);

        try {
            const response = await fetch('/process', {
                method: 'POST',
                body: new FormData(form),
            });
            const data = await response.json();

            if (!response.ok) {
                setStatus(data.error || 'Something went wrong. Please try again.', 'error');
                return;
            }

            setStatus(null);
            renderResults(data.recommendations || []);
            renderRatedRecipes(data.ratedRecipes || [], data.ratedRecipesTotal || 0);
        } catch (err) {
            setStatus('Could not reach the server. Please check your connection and try again.', 'error');
        } finally {
            clearTimeout(coldStartTimer);
            submitBtn.disabled = false;
        }
    });

    userIdInput.focus();
})();
