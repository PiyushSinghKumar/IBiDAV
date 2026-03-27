const state = {
  summary: null,
  selectedCategory: null,
  categoryQuery: "",
  categoryOffset: 0,
  categoryTotal: 0,
  warmupStartedAt: null,
};

function warmupElements() {
  return {
    banner: document.getElementById("warmup-banner"),
    title: document.getElementById("warmup-title"),
    subtitle: document.getElementById("warmup-subtitle"),
    progress: document.getElementById("warmup-progress"),
    track: document.querySelector(".warmup-track"),
  };
}

function updateWarmupBanner(summary) {
  const { banner, title, subtitle, progress, track } = warmupElements();
  if (!banner || !title || !subtitle || !progress || !track) {
    return;
  }

  if (!summary.warming_up) {
    title.textContent = "Initialization complete";
    subtitle.textContent = "Data loaded and ready for search.";
    progress.style.width = "100%";
    track.setAttribute("aria-valuenow", "100");
    window.setTimeout(() => banner.classList.add("hidden"), 900);
    state.warmupStartedAt = null;
    return;
  }

  if (!state.warmupStartedAt) {
    state.warmupStartedAt = Date.now();
  }

  const elapsedSec = Math.max(1, Math.floor((Date.now() - state.warmupStartedAt) / 1000));
  const estimated = Math.min(92, 12 + elapsedSec * 3);
  const initState = summary.init_state || {};
  const hasError = Boolean(initState.error);

  banner.classList.remove("hidden");
  title.textContent = hasError ? "Initialization issue" : "Initializing data";
  subtitle.textContent = hasError
    ? String(initState.error)
    : `${summary.message || "Preparing datasets and indexes..."} (~${elapsedSec}s)`;
  progress.style.width = `${estimated}%`;
  track.setAttribute("aria-valuenow", String(estimated));
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Request failed with status ${response.status}`);
  }
  return response.json();
}

function renderSummary(summary) {
  state.summary = summary;
  const topics = Array.isArray(summary.topics) ? summary.topics : [];
  const categoryCounts = summary.category_counts || {};
  const stats = summary.stats || {};
  const metrics = document.getElementById("summary-metrics");
  metrics.innerHTML = "";
  [
    `${summary.article_rows.toLocaleString()} article-image rows`,
    `${summary.unique_articles.toLocaleString()} unique articles`,
    `${summary.category_count} categories`,
    `${topics.length} cached topics`,
    `${Math.round((stats.multi_label_coverage || 0) * 100)}% labeled coverage`,
  ].forEach((label) => {
    const pill = document.createElement("div");
    pill.className = "metric-pill";
    pill.textContent = label;
    metrics.appendChild(pill);
  });

  const topicChips = document.getElementById("topic-chips");
  topicChips.innerHTML = "";
  topics.forEach((topic) => {
    const chip = document.createElement("div");
    chip.className = "topic-chip";
    chip.textContent = `${topic.label || `Topic ${topic.index + 1}`}: ${topic.words.join(", ")}`;
    topicChips.appendChild(chip);
  });

  const wordcloudImage = document.getElementById("wordcloud-image");
  wordcloudImage.src = summary.wordcloud_base64 ? `data:image/png;base64,${summary.wordcloud_base64}` : "";

  const buttonWrap = document.getElementById("category-buttons");
  buttonWrap.innerHTML = "";
  Object.entries(categoryCounts).forEach(([category, count], index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "category-button";
    button.dataset.category = category;
    button.textContent = `${category} (${count.toLocaleString()})`;
    button.addEventListener("click", () => selectCategory(category));
    if (index === 0) {
      state.selectedCategory = category;
      button.classList.add("active");
    }
    buttonWrap.appendChild(button);
  });

  if (state.selectedCategory) {
    loadCategory(true);
  }
}

function setSearchStatus(message) {
  const el = document.getElementById("search-status");
  el.classList.remove("busy", "error");
  el.textContent = message;
}

function setCategoryStatus(message) {
  const el = document.getElementById("category-status");
  el.classList.remove("busy", "error");
  el.textContent = message;
}

function setBusyStatus(id, message) {
  const el = document.getElementById(id);
  el.classList.remove("error");
  el.classList.add("busy");
  el.textContent = message;
}

function setErrorStatus(id, message) {
  const el = document.getElementById(id);
  el.classList.remove("busy");
  el.classList.add("error");
  el.textContent = message;
}

function articleCard(article) {
  const relevance = Number.isFinite(article.score) ? article.score.toFixed(3) : "N/A";
  return `
    <article class="result-card">
      <div class="result-meta-stack">
        <div class="result-meta">${escapeHtml(article.category ? article.category : "Search Result")}</div>
        <div class="result-meta">PMID ${escapeHtml(article.pmid || "N/A")}</div>
        <div class="result-meta">PMCID ${escapeHtml(article.pmcid || "N/A")}</div>
        <div class="result-meta">Relevance ${escapeHtml(relevance)}</div>
      </div>
      <h3>${escapeHtml(article.title || "Untitled")}</h3>
      <p>${escapeHtml(article.abstract || "No abstract available.")}</p>
      ${article.article_url ? `<a href="${escapeHtml(article.article_url)}" target="_blank" rel="noreferrer">Open article</a>` : ""}
    </article>
  `;
}

function categoryCard(article) {
  const imageHtml = article.image_urls
    .map((url) => `<img src="${escapeHtml(url)}" loading="lazy" alt="Biomedical figure thumbnail" />`)
    .join("");
  return `
    <article class="result-card">
      <div class="result-meta-stack">
        <div class="result-meta">PMID ${escapeHtml(article.pmid || "N/A")}</div>
        <div class="result-meta">PMCID ${escapeHtml(article.pmcid || "N/A")}</div>
        <div class="result-meta">${escapeHtml(article.image_count)} images</div>
      </div>
      <h3>${escapeHtml(article.title || "Untitled")}</h3>
      <p>${escapeHtml(article.abstract_preview || "No abstract available.")}</p>
      ${article.article_url ? `<a href="${escapeHtml(article.article_url)}" target="_blank" rel="noreferrer">Open article</a>` : ""}
      <div class="image-strip">${imageHtml}</div>
    </article>
  `;
}

async function loadSummary() {
  const summary = await fetchJson("/api/summary");
  renderSummary(summary);
  updateWarmupBanner(summary);
  if (summary.warming_up) {
    const note = summary.message || "Data is warming up. Please wait...";
    setBusyStatus("search-status", note);
    setBusyStatus("category-status", note);
    window.setTimeout(loadSummary, 3000);
    return;
  }
  setSearchStatus("Enter a query to search the corpus.");
  setCategoryStatus("Select a category to explore records.");
}

async function runSearch(query) {
  const resultsWrap = document.getElementById("search-results");
  if (!query.trim()) {
    resultsWrap.innerHTML = "";
    setSearchStatus("Enter a query to search the corpus.");
    return;
  }

  setBusyStatus("search-status", "Searching...");
  try {
    const payload = await fetchJson(`/api/search?q=${encodeURIComponent(query)}`);
    if (payload.warming_up) {
      setBusyStatus("search-status", "Data is still warming up. Please retry shortly.");
      return;
    }
    if (!payload.results.length) {
      resultsWrap.innerHTML = "";
      setSearchStatus(`No results found for \"${payload.query}\".`);
      return;
    }
    resultsWrap.innerHTML = payload.results.map(articleCard).join("");
    setSearchStatus(`${payload.results.length.toLocaleString()} ranked results for \"${payload.query}\"`);
  } catch (error) {
    setErrorStatus("search-status", error.message || "Search request failed.");
  }
}

function syncCategoryButtons() {
  document.querySelectorAll(".category-button").forEach((button) => {
    const isActive = button.dataset.category === state.selectedCategory;
    button.classList.toggle("active", Boolean(isActive));
  });
}

function selectCategory(category) {
  state.selectedCategory = category;
  state.categoryOffset = 0;
  syncCategoryButtons();
  loadCategory(true);
}

async function loadCategory(reset) {
  if (!state.selectedCategory) {
    return;
  }
  const resultsWrap = document.getElementById("category-results");
  const offset = reset ? 0 : state.categoryOffset;
  const limit = reset ? pageConfig.categoryPageSize : pageConfig.loadMorePageSize;
  setBusyStatus("category-status", `Loading ${state.selectedCategory}...`);

  try {
    const payload = await fetchJson(
      `/api/categories/${encodeURIComponent(state.selectedCategory)}?q=${encodeURIComponent(state.categoryQuery)}&offset=${offset}&limit=${limit}`
    );
    if (payload.warming_up) {
      setBusyStatus("category-status", "Category data is warming up. Please retry shortly.");
      document.getElementById("load-more-button").classList.add("hidden");
      return;
    }

    if (reset) {
      resultsWrap.innerHTML = payload.results.map(categoryCard).join("");
    } else {
      resultsWrap.insertAdjacentHTML("beforeend", payload.results.map(categoryCard).join(""));
    }

    state.categoryOffset = payload.next_offset;
    state.categoryTotal = payload.total;
    setCategoryStatus(`Showing ${state.categoryOffset.toLocaleString()} of ${payload.total.toLocaleString()} results in ${payload.category}`);
    document.getElementById("load-more-button").classList.toggle("hidden", payload.remaining === 0);
  } catch (error) {
    setErrorStatus("category-status", error.message || "Category request failed.");
  }
}

document.getElementById("search-form").addEventListener("submit", (event) => {
  event.preventDefault();
  runSearch(document.getElementById("search-input").value);
});

document.getElementById("category-search-form").addEventListener("submit", (event) => {
  event.preventDefault();
  state.categoryQuery = document.getElementById("category-search-input").value;
  state.categoryOffset = 0;
  loadCategory(true);
});

document.getElementById("load-more-button").addEventListener("click", () => loadCategory(false));

loadSummary().catch((error) => {
  const { banner, title, subtitle } = warmupElements();
  if (banner && title && subtitle) {
    banner.classList.remove("hidden");
    title.textContent = "Initialization issue";
    subtitle.textContent = error.message || "Failed to load summary.";
  }
  setErrorStatus("search-status", error.message || "Failed to load summary.");
  setErrorStatus("category-status", error.message || "Failed to load summary.");
});
