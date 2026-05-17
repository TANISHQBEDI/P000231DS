import { useMemo } from "react";
import { DataGrid, renderTextEditor } from "react-data-grid";
import "react-data-grid/lib/styles.css";

// Generic editable grid. `columns` is a string[] of field names; every
// column is rendered as an inline-editable text cell.
function EditableGrid({ columns, rows, onRowsChange }) {
  const gridColumns = useMemo(
    () =>
      (columns || []).map((c) => ({
        key: c,
        name: c,
        editable: true,
        resizable: true,
        renderEditCell: renderTextEditor,
      })),
    [columns],
  );

  if (!rows?.length) return null;

  return (
    <div className="grid-wrap">
      <DataGrid
        columns={gridColumns}
        rows={rows}
        onRowsChange={onRowsChange}
        rowKeyGetter={(r) => r.id}
        className="rdg-light"
        style={{ blockSize: Math.min(60 + rows.length * 35, 480) }}
      />
    </div>
  );
}

export default EditableGrid;
