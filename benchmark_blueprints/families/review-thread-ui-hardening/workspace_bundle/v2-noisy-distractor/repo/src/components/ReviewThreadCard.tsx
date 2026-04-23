export function renderReviewThreadCards() {
  return `
    <article class="review-thread-card" data-thread-state="unresolved">
      <div class="review-thread-card__meta-row">
        <p class="review-thread-card__title">
          Composer footer wraps badly on mobile when the reply menu sits beside a long reviewer summary.
        </p>
        <button class="review-thread-card__menu-button" data-control="reply-thread-menu" type="button">
          <svg aria-hidden="true" viewBox="0 0 16 16"></svg>
        </button>
      </div>
    </article>
    <article class="review-thread-card" data-thread-state="resolved">
      <div class="review-thread-card__meta-row">
        <p class="review-thread-card__title">Resolved archive note</p>
        <button class="review-thread-card__menu-button" data-control="resolved-thread-menu" type="button">
          <svg aria-hidden="true" viewBox="0 0 16 16"></svg>
        </button>
      </div>
    </article>
  `;
}
