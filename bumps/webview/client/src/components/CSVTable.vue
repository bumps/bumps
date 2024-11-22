<script setup lang="ts">
import { ref } from 'vue';

export type TableData = {
    raw: string,
    header: string[],
    rows: string[][]
}

const hidden_download = ref<HTMLAnchorElement>();

const props = defineProps<{table_data: TableData}>();

async function download_csv() {
  if (props.table_data.raw) {
    const a = hidden_download.value as HTMLAnchorElement;
    a.href = 'data:text/csv;charset=utf-8,' + encodeURIComponent(props.table_data.raw);
    a.click();
  }
}

</script>

<template>
    <div class="flex-grow-0" ref="table_div">
        <div>
            <button class="btn btn-primary btn-sm" @click="download_csv">Download CSV</button>
            <a ref="hidden_download" class="hidden" download='table.csv' type='text/csv'>Download CSV</a>
        </div>
        <table class="table">
            <thead class="border-bottom py-1 sticky-top text-white bg-secondary">
                <tr>
                    <th v-for="header_item in table_data.header" scope="col">{{ header_item }}</th>
                </tr>
            </thead>
            <tbody>
                <tr class="py-1" v-for="table_row in table_data.rows">
                    <td v-for="table_item in table_row" scope="col">{{ table_item }}</td>
                </tr>
            </tbody>
        </table>
    </div>
  </template>

<style scoped>
.hidden {
  display: none;
}
</style>