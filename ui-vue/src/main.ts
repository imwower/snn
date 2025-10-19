import { createApp } from 'vue';
import { createPinia } from 'pinia';
import App from './App.vue';
import './styles.css';
import { startSocket } from './ws';

const pinia = createPinia();
startSocket(pinia);

const app = createApp(App);
app.use(pinia);
app.mount('#app');
