import { ref, onMounted, onBeforeUnmount } from 'vue';
import type { AsyncSocket } from './asyncSocket';

type Message = {
  timestamp: string,
  message: object
}

export function setupDrawLoop(topic: string, socket: AsyncSocket, draw: Function, name: string = '') {
  const mounted = ref(false);
  const drawing_busy = ref(false);
  const draw_requested = ref(false);
  const latest_timestamp = ref<string>();

  const topic_callback = function(message) {
    const new_timestamp = message?.timestamp;
    if (new_timestamp !== undefined) {
      latest_timestamp.value = new_timestamp;
    }
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

    const messages = await socket.asyncEmit('get_topic_messages', topic) as Message[];
    // console.log(topic, messages);
    const last_message = messages.pop();
    if (last_message !== undefined) {
      topic_callback(last_message);
    }
    window.requestAnimationFrame(draw_if_needed);
  });

  onBeforeUnmount(() => { 
    mounted.value = false;
    socket.off(topic, topic_callback);
  });

  return { mounted, drawing_busy, draw_requested };
}
