## 📘 Aura AI – AI-Powered Tutoring Platform (Frontend)
Aura AI is a professional-grade AI-powered tutoring platform designed to connect students with culturally-aware expert tutors from around the world. This frontend is crafted in pure HTML, CSS, and JavaScript, offering a responsive, modern UI with interactive components for learners and tutors.
 
## 🌐 Live Preview

Once deployed: `https://aura-ai-kj2k.vercel.app/#`

## 🚀 Features
Stunning Hero Section: Engaging welcome area with calls to action.
AI Matching System (UI): Visual design for tutor matching powered by intelligent algorithms.
Cultural Intelligence Highlights: Tailored UI showcasing Aura AI’s inclusive approach.
Progress Analytics UI: UI elements for learning tracking and personalized feedback.
Flexible Session Scheduling: Interface for 24/7 tutor session booking.
Secure & Verified Tutor Profiles: UI elements for security, background-verified tutors.
Simple Pricing Models: Pricing cards with dynamic plan highlights.
Contact & Support: Responsive contact form and contact information layout.
Fully Responsive: Mobile-friendly and optimized for all devices.
Interactive Navigation: Page routing using JavaScript (no reload).
User Menu / Authentication Buttons: Ready UI elements for login/register and dashboard profile.

## 📁 File Structure
aura_professional_frontend.html
The entire frontend is encapsulated in a single HTML file with embedded CSS and JavaScript for demo or rapid prototyping purposes.

## 🖥️ Technologies Used
HTML5
CSS3 (Custom Variables + Utility Styles)
JavaScript (Vanilla)
Font Awesome 6
Google Fonts - Inter

## 📸 UI Previews
Hero Section with CTA Buttons
Dynamic Navigation Bar with Scroll Sticky Header
Features Grid with Icons
Testimonials Carousel
“How It Works” Steps Layout
Pricing Plans Cards
Contact Form with Notification
Footer with Social Links

## 🔐 Ready for Integration
This UI is backend-ready. Simply connect to your API endpoints via the provided structure using the apiCall() function. Authentication tokens are already managed through localStorage and passed via headers.

function apiCall(endpoint, options = {}) {
    ...
    config.headers.Authorization = `Bearer ${authToken}`;
    return fetch(`${API_BASE}${endpoint}`, config);
}

## ⚙️ Setup Instructions
Clone the repository or copy the HTML file.
Open aura_professional_frontend.html in your browser.
Optionally update API_BASE to your backend URL.
Connect backend endpoints as needed.

## 🧪 Upcoming Enhancements
✅ Dynamic Tutor Listings
✅ Profile Editing Page
✅ Full Dashboard Integration
✅ Admin Panel UI
✅ Dark Mode Support
📄 License
This project is licensed under the MIT License – feel free to use and modify for commercial or personal projects.



