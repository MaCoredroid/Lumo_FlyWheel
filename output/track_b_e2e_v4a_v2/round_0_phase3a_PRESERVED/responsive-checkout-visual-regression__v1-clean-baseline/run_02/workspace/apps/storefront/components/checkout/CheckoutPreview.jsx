export function CheckoutPreview({ previewCompactSummary = true }) {
  return (
    <main className="checkout-shell" data-preview-compact={previewCompactSummary ? "true" : "false"}>
      <section className="checkout-payment" aria-label="Payment details">
        <h1>Checkout</h1>
        <label>
          Card number
          <input aria-label="Card number" placeholder="4242 4242 4242 4242" />
        </label>
        <label>
          Billing ZIP
          <input aria-label="Billing ZIP" placeholder="94107" />
        </label>
        <button className="checkout-primary-cta">Place order</button>
      </section>
      <aside className="checkout-summary" aria-label="Order summary">
        <h2>Order summary</h2>
        <p>Subtotal: $84.00</p>
        <p>Shipping: $6.00</p>
        <strong>Total: $90.00</strong>
      </aside>
      <footer className="checkout-mobile-bar">
        <span>Total $90.00</span>
        <button>Review order</button>
      </footer>
    </main>
  );
}
