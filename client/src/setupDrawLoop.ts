import { ref, onMounted, onBeforeUnmount } from 'vue';
import type { Socket } from 'socket.io-client';
import './asyncSocket';

export function setupDrawLoop(topic: string, socket: Socket, draw: Function, name: string = '') {
  const mounted = ref(false);
  const drawing_busy = ref(false);
  const draw_requested = ref(false);
  const latest_timestamp = ref<string>();

  const topic_callback = function() {
    draw_requested.value = true;
  }

  const draw_if_needed = async function() {
    if (!mounted.value) {
      return;
    }
    if (drawing_busy.value) {
      console.log(`drawing ${name} busy!`);
    }
    else if (draw_requested.value) {
      drawing_busy.value = true;
      draw_requested.value = false;
      await draw(latest_timestamp.value);
      drawing_busy.value = false;
    }
    window.requestAnimationFrame(draw_if_needed);
  }

  onMounted(async () => {
    mounted.value = true;
    socket.on(topic, topic_callback);

    const messages = await socket.asyncEmit('get_topic_messages', topic);
    // console.log(topic, messages);
    const last_message = messages.pop();
    if (last_message) {
      draw_requested.value = true;
      latest_timestamp.value = last_message.timestamp;
    }
    window.requestAnimationFrame(draw_if_needed);
  });

  onBeforeUnmount(() => { 
    mounted.value = false;
    socket.off(topic, topic_callback);
  });

  return { mounted, drawing_busy, draw_requested };
}
