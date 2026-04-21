/// <reference types="vite/client" />

// Allow importing .wgsl files as raw strings
declare module '*.wgsl?raw' {
  const content: string;
  export default content;
}
