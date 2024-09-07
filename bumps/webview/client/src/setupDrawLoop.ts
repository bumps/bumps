import { ref, onActivated, onDeactivated } from 'vue';
import type { AsyncSocket } from './asyncSocket.ts';

type Message = {
  timestamp: string,
  message: object
}

export function setupDrawLoop(topic: string, socket: AsyncSocket, draw: Function, name: string = '') {
  const mounted = ref(false);
  const drawing_busy = ref(false);
  const draw_requested = ref(false);
  const latest_value = ref<unknown>();

  const topic_callback = function(value: unknown) {
    latest_value.value = value;
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
      try {
        // Need to continue the draw loop even if draw fails
        await draw(latest_value.value);
      }
      catch (e) {
        // TODO: should this notify the user?
        console.error(`Error drawing ${name}:`, e);
        // add sleep to avoid runaway error loop
        await sleep(1000);
      }
      drawing_busy.value = false;
    }
    window.requestAnimationFrame(draw_if_needed);
  }

  const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

  onActivated(async () => {
    mounted.value = true;
    socket.on(topic, topic_callback);
    const value = await socket.asyncEmit('get_shared_setting', topic);
    if (value !== undefined) {
      topic_callback(value);
    }
    window.requestAnimationFrame(draw_if_needed);
  });

  onDeactivated(() => {
    mounted.value = false;
    socket.off(topic, topic_callback);
  });

  return { mounted, drawing_busy, draw_requested };
}
