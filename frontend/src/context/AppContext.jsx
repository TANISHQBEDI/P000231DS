// Lightweight shared state so navigating the sidebar doesn't lose work.

import { createContext, useContext, useMemo, useState } from "react";

const AppContext = createContext(null);

export function AppProvider({ children }) {
  const [uploadedRows, setUploadedRows] = useState([]);
  const [columns, setColumns] = useState([]);
  const [editedRows, setEditedRows] = useState([]);
  const [prediction, setPrediction] = useState(null);

  const value = useMemo(
    () => ({
      uploadedRows,
      setUploadedRows,
      columns,
      setColumns,
      editedRows,
      setEditedRows,
      prediction,
      setPrediction,
    }),
    [uploadedRows, columns, editedRows, prediction],
  );

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

export function useApp() {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error("useApp must be used within an AppProvider");
  return ctx;
}
