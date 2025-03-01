declare module 'vitest/config' {
  export { defineConfig } from 'vitest';
}

declare module '@vitejs/plugin-react' {
  const plugin: any;
  export default plugin;
} 