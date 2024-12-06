<script setup lang="ts">
import { ref } from "vue";

export type TableData = {
  raw: string;
  header: string[];
  rows: string[][];
};

const hidden_download = ref<HTMLAnchorElement>();

const props = defineProps<{ tableData: TableData }>();

async function download_csv() {
  if (props.tableData.raw) {
    const a = hidden_download.value as HTMLAnchorElement;
    a.href = "data:text/csv;charset=utf-8," + encodeURIComponent(props.tableData.raw);
    a.click();
  }
}
</script>

<template>
  <div ref="table_div" class="flex-grow-0">
    <div>
      <button class="btn btn-primary btn-sm" @click="download_csv">Download CSV</button>
      <a ref="hidden_download" class="hidden" download="table.csv" type="text/csv">Download CSV</a>
    </div>
    <table class="table">
      <thead class="border-bottom py-1 sticky-top text-white bg-secondary">
        <tr>
          <th v-for="header_item in tableData.header" :key="`header-${header_item}`" scope="col">{{ header_item }}</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="(table_row, table_row_index) in tableData.rows" :key="`row-${table_row_index}`" class="py-1">
          <td v-for="(table_item, table_item_index) in table_row" :key="`item-${table_item_index}`" scope="col">
            {{ table_item }}
          </td>
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
